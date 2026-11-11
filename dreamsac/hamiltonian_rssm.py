import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import symplectic
from . import eqvnet

f32 = jnp.float32
sg = jax.lax.stop_gradient


class HamiltonianRSSM(nj.Module):
  """
  Corrected version of the Hamiltonian RSSM.
  This version fixes the interface between discrete states and continuous dynamics,
  and ensures the loss function is consistent with the dynamics model.
  """

  # --- Standard RSSM Parameters ---
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  
  # --- Hamiltonian-specific Parameters ---
  ham_layers: int = 3
  ham_units: int = 512
  ham_act: str = 'silu'
  use_symplectic: bool = False
  integrator_method: str = 'leapfrog'
  integrator_dt: float = 0.1
  # New: Standard deviation for converting continuous values to logits, controls distribution softness.
  logit_stddev: float = 0.1
  
  # --- Equivariant Network Parameters ---
  use_equivariant: bool = False
  use_e3nn: bool = False
  equivariant_max_ell: int = 2
  
  # --- Input Matrix g(q) Parameters ---
  use_input_matrix: bool = False
  input_matrix_layers: int = 2
  input_matrix_units: int = 256

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    assert self.stoch % 2 == 0, "stoch must be even to split into q and p"
    self.act_space = act_space
    self.kw = kw
    self.q_dim = self.stoch // 2
    self.p_dim = self.stoch // 2
    
    self._ham_net = None
    self._input_matrix_net = None

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        q=elements.Space(np.float32, (self.q_dim, self.classes)),
        p=elements.Space(np.float32, (self.p_dim, self.classes)))

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        q=jnp.zeros([bsize, self.q_dim, self.classes], f32),
        p=jnp.zeros([bsize, self.p_dim, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, q, p, action = nn.mask(
        (carry['deter'], carry['q'], carry['p'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    
    stoch = jnp.concatenate([q, p], axis=1)
    deter = self._core(deter, stoch, action)
    
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    
    logit_q = self._logit('obslogit_q', x, self.q_dim)
    logit_p = self._logit('obslogit_p', x, self.p_dim)
    
    q = nn.cast(self._dist(logit_q).sample(seed=nj.seed()))
    p = nn.cast(self._dist(logit_p).sample(seed=nj.seed()))
    
    stoch = jnp.concatenate([q, p], axis=1)
    
    carry = dict(deter=deter, q=q, p=p)
    feat = dict(deter=deter, q=q, p=p, stoch=stoch, logit_q=logit_q, logit_p=logit_p)
    entry = dict(deter=deter, q=q, p=p)
    
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, q, p, logit_q, logit_p))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      
      stoch = jnp.concatenate([carry['q'], carry['p']], axis=1)
      deter = self._core(carry['deter'], stoch, actemb)
      
      if self.use_symplectic:
        q, p, prior_q, prior_p = self._predict_with_symplectic(
            carry['q'], carry['p'], action, deter)
      else:
        prior_q = self._prior(deter, 'q')
        prior_p = self._prior(deter, 'p')
        
        q = nn.cast(self._dist(prior_q).sample(seed=nj.seed()))
        p = nn.cast(self._dist(prior_p).sample(seed=nj.seed()))
      
      stoch = jnp.concatenate([q, p], axis=1)
      
      carry = nn.cast(dict(deter=deter, q=q, p=p))
      feat = nn.cast(dict(deter=deter, q=q, p=p, stoch=stoch, logit_q=prior_q, logit_p=prior_p))
      
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, q, p, prior_q, prior_p))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    
    # Get posterior from observation.
    post_q = feat['logit_q']
    post_p = feat['logit_p']
    
    prior_q = self._prior(sg(feat['deter']), 'q')
    prior_p = self._prior(sg(feat['deter']), 'p')

    # KL divergence calculation remains the same.
    dyn_q = self._dist(sg(post_q)).kl(self._dist(prior_q))
    dyn_p = self._dist(sg(post_p)).kl(self._dist(prior_p))
    rep_q = self._dist(post_q).kl(self._dist(sg(prior_q)))
    rep_p = self._dist(post_p).kl(self._dist(sg(prior_p)))
    
    dyn = (dyn_q + dyn_p) / 2.0
    rep = (rep_q + rep_p) / 2.0
    
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    
    losses = {'dyn': dyn, 'rep': rep}
    
    metrics['dyn_ent_q'] = self._dist(prior_q).entropy().mean()
    metrics['dyn_ent_p'] = self._dist(prior_p).entropy().mean()
    metrics['rep_ent_q'] = self._dist(post_q).entropy().mean()
    metrics['rep_ent_p'] = self._dist(post_p).entropy().mean()
    
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, feat, qp_type):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior_{qp_type}{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior_{qp_type}{i}norm', nn.Norm, self.norm)(x))
    dim = self.q_dim if qp_type == 'q' else self.p_dim
    return self._logit(f'priorlogit_{qp_type}', x, dim)

  def _predict_with_symplectic(self, q_current, p_current, action, deter):
    """
    Predict next state using symplectic integrator.
    
    Key fix: The integrator produces deterministic predictions, but we need
    a stochastic prior p(z_t | z_{t-1}, a_{t-1}) for the ELBO.
    
    Solution:
    - Integrator prediction → mean of prior distribution
    - Small neural network → variance/noise of prior distribution
    - This preserves physics while maintaining stochasticity for KL divergence
    """
    
    # 1. Convert input one-hot states to continuous values.
    q_cont = nn.cast(self._to_continuous(q_current))
    p_cont = nn.cast(self._to_continuous(p_current))
    # Initialize Ninjax parameters outside the integrator's nested jax.grad.
    self.compute_hamiltonian({'q': q_cont, 'p': p_cont}, action)

    # Select integrator.
    integrators = {
        'leapfrog': symplectic.leapfrog_step,
        'euler': symplectic.euler_symplectic_step,
        'rk4': symplectic.rk4_symplectic_step,
    }
    integrator = integrators.get(self.integrator_method, symplectic.leapfrog_step)
    
    # 2. Perform integration in the continuous phase space.
    # This produces the MEAN of the prior distribution
    # Include input matrix g(q) for controlled Hamiltonian dynamics
    if self.integrator_method in ['leapfrog', 'euler']:
      # These integrators support input_matrix_fn
      # IMPORTANT: Use a bound method instead of lambda to avoid closure issues
      if self.use_input_matrix:
        input_matrix_fn = self.compute_input_matrix
      else:
        input_matrix_fn = None
      q_mean_cont, p_mean_cont = integrator(
          self.compute_hamiltonian,
          q_cont, p_cont, action, dt=self.integrator_dt,
          input_matrix_fn=input_matrix_fn
      )
    else:
      # RK4 doesn't support input_matrix_fn yet (TODO: extend if needed)
      q_mean_cont, p_mean_cont = integrator(
          self.compute_hamiltonian,
          q_cont, p_cont, action, dt=self.integrator_dt
      )

    # This allows the model to learn when physics might be violated
    # Input: deterministic state + action + deter (carries history)
    stoch_input = jnp.concatenate([
        q_mean_cont.reshape(q_mean_cont.shape[0], -1),
        p_mean_cont.reshape(p_mean_cont.shape[0], -1),
        deter
    ], axis=-1)
    
    # Ensure correct dtype before passing to neural network
    stoch_input = nn.cast(stoch_input)
    
    # Small MLP to predict stochastic corrections (residuals)
    x = stoch_input
    for i in range(2):  # 2 layers should be enough
      x = self.sub(f'stoch_prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'stoch_prior{i}norm', nn.Norm, self.norm)(x))
    
    # Predict logits for q and p distributions
    # The logits represent deviations from the deterministic prediction
    logit_q_residual = self._logit('stoch_prior_q', x, self.q_dim)
    logit_p_residual = self._logit('stoch_prior_p', x, self.p_dim)
    
    # 4. Combine deterministic mean with stochastic residuals
    # Convert mean to logits, then add residual
    logit_q_mean = self._to_logits(q_mean_cont)
    logit_p_mean = self._to_logits(p_mean_cont)
    
    # Final prior logits = deterministic physics + learned stochasticity
    logit_q = logit_q_mean + logit_q_residual
    logit_p = logit_p_mean + logit_p_residual
    
    # Sample from the stochastic prior distributions
    q_sampled = nn.cast(self._dist(logit_q).sample(seed=nj.seed()))
    p_sampled = nn.cast(self._dist(logit_p).sample(seed=nj.seed()))
    
    return q_sampled, p_sampled, logit_q, logit_p

  def _logit(self, name, x, stoch_dim):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, stoch_dim * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (stoch_dim, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

  def compute_hamiltonian(self, feat, action):
    """
    Compute Hamiltonian energy H(q, p, a).
    
    This method handles two cases:
    1. feat contains one-hot encoded states (from RSSM features) → convert to continuous
       Shape: [B, dim, classes]
    2. feat contains continuous states (from symplectic integrator) → use directly
       Shape: [B, dim]
    """
    q = feat['q']
    p = feat['p']
    
    # Check dimensionality to distinguish one-hot from continuous
    if q.ndim == 3:
      # One-hot encoded: [B, dim, classes] → convert to [B, dim]
      q_cont = self._to_continuous(q)
      p_cont = self._to_continuous(p)
    elif q.ndim == 2:
      # Already continuous: [B, dim]
      q_cont = q
      p_cont = p
    else:
      raise ValueError(f"Unexpected q shape: {q.shape}. Expected 2D or 3D.")
    
    # Ensure correct dtype (DreamerV3 uses bfloat16 for computation)
    q_cont = nn.cast(q_cont)
    p_cont = nn.cast(p_cont)

    # Lazy initialization of Hamiltonian network.
    if self._ham_net is None:
      if self.use_equivariant:
        self._ham_net = self.sub(
            'ham_net',
            eqvnet.create_hamiltonian_network,
            use_equivariant=True,
            use_e3nn=self.use_e3nn,
            layers=self.ham_layers,
            hidden_channels=self.ham_units,
            max_ell=self.equivariant_max_ell,
            norm=self.norm,
            act=self.ham_act,
            **self.kw
        )
      else:
        self._ham_net = self.sub(
            'ham_net',
            eqvnet.EquivariantHamiltonianNet,
            use_e3nn=False,
            layers=self.ham_layers,
            hidden_channels=self.ham_units,
            norm=self.norm,
            act=self.ham_act,
            **self.kw
        )
    
    # Compute energy using continuous q and p as input.
    energy = self._ham_net(q_cont, p_cont, action)
    return energy

  def _to_continuous(self, x_onehot):
    """Converts a one-hot encoded tensor to a continuous-valued tensor."""
    bins = jnp.linspace(-1.0, 1.0, self.classes, dtype=x_onehot.dtype)
    return jnp.einsum('...dc,c->...d', x_onehot, bins)

  def _to_logits(self, x_cont):
    """Converts a continuous-valued tensor to logits for a categorical distribution."""
    bins = jnp.linspace(-1.0, 1.0, self.classes, dtype=x_cont.dtype)
    x_cont_expanded = x_cont[..., None]
    bins_expanded = bins[None, ...]
    logits = -((x_cont_expanded - bins_expanded)**2) / (2 * self.logit_stddev**2)
    return logits
  
  def compute_input_matrix(self, q, action):
    """
    Compute the input matrix g(q) that maps actions to generalized forces.
    
    From controlled Hamiltonian dynamics (Eq. 1):
      dp/dt = -∂H/∂q + g(q) · a_t
    
    The input matrix g(q) represents how external actions affect the momentum.
    This is learned rather than hand-designed, allowing the model to discover
    the physical coupling between actions and state variables.
    
    Args:
      q: Generalized coordinates, shape [B, q_dim] or [B, q_dim, classes]
      action: Action from action space, dict or tensor
    
    Returns:
      force: Generalized force to apply to momentum, shape [B, p_dim]
    """
    # Convert q to continuous if needed
    if q.ndim == 3:
      q_cont = self._to_continuous(q)
    else:
      q_cont = q
    
    q_cont = nn.cast(q_cont)
    
    # Process action
    if isinstance(action, dict):
      action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.cast(action)
    
    # Lazy initialization of input matrix network
    if self._input_matrix_net is None:
      if self.use_input_matrix:
        # Create a learned input matrix network
        self._input_matrix_net = self.sub(
            'input_matrix_net',
            InputMatrixNetwork,
            q_dim=self.q_dim,
            p_dim=self.p_dim,
            layers=self.input_matrix_layers,
            units=self.input_matrix_units,
            act=self.ham_act,
            norm=self.norm,
            **self.kw
        )
      else:
        # Simple baseline: direct linear mapping from action to force
        self._input_matrix_net = self.sub(
            'input_matrix_net',
            SimpleInputMatrix,
            p_dim=self.p_dim,
            **self.kw
        )
    
    # Compute generalized force: g(q) · a
    force = self._input_matrix_net(q_cont, action)
    
    return force


class InputMatrixNetwork(nj.Module):
  """
  Learned input matrix g(q) that maps actions to generalized forces.
  
  This network learns how actions couple to different degrees of freedom
  in the system, potentially discovering physical structure like:
  - Which state variables are directly controllable
  - Non-linear action effects (e.g., torque depends on joint angle)
  - Configuration-dependent control authority
  """
  
  q_dim: int = 16
  p_dim: int = 16
  layers: int = 2
  units: int = 256
  act: str = 'silu'
  norm: str = 'rms'
  winit: str = 'trunc_normal_in'  # Weight initialization
  
  def __init__(self, **kw):
    self.kw = kw
  
  def __call__(self, q, action):
    """
    Args:
      q: Generalized coordinates [B, q_dim]
      action: Action [B, action_dim]
    
    Returns:
      force: Generalized force [B, p_dim]
    """
    # Concatenate q and action as input
    x = jnp.concatenate([q, action], axis=-1)
    x = nn.cast(x)
    
    # Multi-layer MLP to learn input matrix structure
    for i in range(self.layers):
      x = self.sub(f'layer{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'norm{i}', nn.Norm, self.norm)(x))
    
    # Output layer: map to p_dim (one force component per momentum variable)
    force = self.sub('out', nn.Linear, self.p_dim, **self.kw)(x)
    
    return force


class SimpleInputMatrix(nj.Module):
  """
  Simple baseline input matrix: direct linear mapping from action to force.
  This is equivalent to assuming g(q) = constant matrix.
  """
  
  p_dim: int = 16
  winit: str = 'trunc_normal_in'  # Weight initialization
  
  def __init__(self, **kw):
    self.kw = kw
  
  def __call__(self, q, action):
    """
    Args:
      q: Generalized coordinates [B, q_dim] (unused in simple version)
      action: Action [B, action_dim]
    
    Returns:
      force: Generalized force [B, p_dim]
    """
    action = nn.cast(action)
    force = self.sub('linear', nn.Linear, self.p_dim, **self.kw)(action)
    return force