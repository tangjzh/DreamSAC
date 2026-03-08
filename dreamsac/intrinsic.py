import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn

f32 = jnp.float32


def compute_intrinsic_reward_for_imagination(hamiltonian_fn, imgfeat, imgact):
  """
  Compute intrinsic rewards for imagined trajectories based on energy conservation violation.
  
  Formula: r_t = |H(z_t, a_t) - H(z_{t-1}, a_{t-1})|
  
  This formula has clear physical meaning:
  - At time t-1: system is in state z_{t-1}, agent applies action a_{t-1}, 
                 total energy is H(z_{t-1}, a_{t-1})
  - At time t: system is in state z_t, agent applies action a_t,
               total energy is H(z_t, a_t)
  - The energy change |ΔH| indicates how well the model's physics captures reality
  
  Args:
    hamiltonian_fn: Function that computes H(feat, action) → scalar energy
    imgfeat: Imagined features dict with shape [B*K, H+1, ...]
             Contains 'q', 'p', 'deter' for each timestep
    imgact: Imagined actions with shape [B*K, H+1, ...]
  
  Returns:
    Intrinsic rewards with shape [B*K, H+1]
    Note: First timestep reward is 0 (no previous state to compare)
  """
  BK, H1 = imgfeat['deter'].shape[:2]  # H1 = H + 1
  
  # Compute energy H(z_t, a_t) for each timestep
  energies = []
  for t in range(H1):
    feat_t = jax.tree.map(lambda x: x[:, t], imgfeat)
    action_t = jax.tree.map(lambda x: x[:, t], imgact)
    energy_t = hamiltonian_fn(feat_t, action_t)  # [B*K]
    energies.append(energy_t)
  
  energies = jnp.stack(energies, axis=1)  # [B*K, H+1]
  
  # Compute energy difference: r_t = |H(z_t, a_t) - H(z_{t-1}, a_{t-1})|
  # energies[:, 1:] contains H(z_1, a_1), H(z_2, a_2), ..., H(z_H, a_H)
  # energies[:, :-1] contains H(z_0, a_0), H(z_1, a_1), ..., H(z_{H-1}, a_{H-1})
  energy_changes = jnp.abs(energies[:, 1:] - energies[:, :-1])  # [B*K, H]
  
  # Pad first timestep with zeros (no previous state to compare)
  intrinsic_rewards = jnp.concatenate([
      jnp.zeros((BK, 1)), energy_changes], axis=1)  # [B*K, H+1]
  
  return intrinsic_rewards


def compute_intrinsic_reward_for_observation(hamiltonian_fn, obs_feat_seq, obs_action_seq):
  """
  Compute intrinsic rewards for observed trajectories (used during data collection).
  
  Args:
    hamiltonian_fn: Function that computes H(feat, action) → scalar energy
    obs_feat_seq: Observed features dict with shape [B, T, ...]
    obs_action_seq: Observed actions with shape [B, T, ...]
  
  Returns:
    Intrinsic rewards with shape [B, T-1]
  """
  B, T = obs_feat_seq['deter'].shape[:2]
  
  # Compute energy H(z_t, a_t) for each timestep
  energies = []
  for t in range(T):
    feat_t = jax.tree.map(lambda x: x[:, t], obs_feat_seq)
    action_t = jax.tree.map(lambda x: x[:, t], obs_action_seq)
    energy_t = hamiltonian_fn(feat_t, action_t)  # [B]
    energies.append(energy_t)
  
  energies = jnp.stack(energies, axis=1)  # [B, T]
  
  # Compute energy difference
  energy_changes = jnp.abs(energies[:, 1:] - energies[:, :-1])  # [B, T-1]
  
  return energy_changes


def compute_energy_conservation_error(hamiltonian_fn, feat_seq, action_seq):
  """
  Compute the total energy conservation error over a trajectory.
  Useful as a diagnostic metric to monitor how well physics is preserved.
  
  Args:
    hamiltonian_fn: Function that computes H(feat, action)
    feat_seq: Features with shape [B, T, ...]
    action_seq: Actions with shape [B, T, ...]
  
  Returns:
    mean_error: Average energy change magnitude [scalar]
    max_error: Maximum energy change magnitude [scalar]
  """
  energy_changes = compute_intrinsic_reward_for_observation(
      hamiltonian_fn, feat_seq, action_seq)
  
  mean_error = jnp.mean(energy_changes)
  max_error = jnp.max(energy_changes)
  
  return mean_error, max_error


def adaptive_intrinsic_reward_scale(energy_changes, target_mean=1.0):
  """
  Compute an adaptive scaling factor to normalize intrinsic rewards.
  
  This prevents the intrinsic reward from dominating or being negligible
  compared to external rewards during fine-tuning.
  
  Args:
    energy_changes: Computed energy changes [B, T]
    target_mean: Desired mean value for normalized rewards
  
  Returns:
    scale: Scalar to multiply with energy_changes
  """
  current_mean = jnp.mean(energy_changes) + 1e-8
  scale = target_mean / current_mean
  
  # Clip to prevent extreme scaling
  scale = jnp.clip(scale, 0.01, 100.0)
  
  return scale


# ============================================================================
# RND (Random Network Distillation) for Novelty-Based Exploration
# ============================================================================

class RNDNetwork(nj.Module):
  """
  Random Network Distillation for computing novelty-based intrinsic rewards.
  
  RND consists of two networks:
  1. Target network: Randomly initialized and fixed (never trained)
  2. Predictor network: Trained to predict the target network's output
  
  The prediction error serves as a novelty signal: high error indicates
  the state is unfamiliar, low error indicates it has been seen often.
  
  Reference: Burda et al., "Exploration by Random Network Distillation", 2018
  """
  
  layers: int = 3
  units: int = 512
  act: str = 'silu'
  norm: str = 'rms'
  output_dim: int = 128
  
  def __init__(self, **kw):
    self.kw = kw
    self._target_net = None
    self._predictor_net = None
  
  def __call__(self, feat, training=False):
    """
    Compute RND prediction error as intrinsic reward.
    
    Args:
      feat: State features (dict with 'deter', 'stoch', etc.)
      training: If True, return both prediction and target for training loss
    
    Returns:
      If training=False: prediction error (intrinsic reward) [B]
      If training=True: (prediction, target, prediction_error)
    """
    # Concatenate all feature components
    if isinstance(feat, dict):
      feat_list = []
      if 'deter' in feat:
        feat_list.append(feat['deter'])
      if 'stoch' in feat:
        feat_list.append(feat['stoch'].reshape(feat['stoch'].shape[0], -1))
      elif 'q' in feat and 'p' in feat:
        q = feat['q'].reshape(feat['q'].shape[0], -1)
        p = feat['p'].reshape(feat['p'].shape[0], -1)
        feat_list.append(jnp.concatenate([q, p], axis=-1))
      x = jnp.concatenate(feat_list, axis=-1)
    else:
      x = feat
    
    x = nn.cast(x)
    
    # Lazy initialization
    if self._target_net is None:
      self._target_net = self.sub('target', MLPNetwork,
                                   layers=self.layers,
                                   units=self.units,
                                   output_dim=self.output_dim,
                                   act=self.act,
                                   norm=self.norm,
                                   **self.kw)
      
      self._predictor_net = self.sub('predictor', MLPNetwork,
                                      layers=self.layers,
                                      units=self.units,
                                      output_dim=self.output_dim,
                                      act=self.act,
                                      norm=self.norm,
                                      **self.kw)
    
    # Target network: stop gradient to keep it fixed
    target = jax.lax.stop_gradient(self._target_net(x))
    
    # Predictor network: trained to match target
    prediction = self._predictor_net(x)
    
    # Prediction error = intrinsic reward (novelty)
    error = jnp.mean((prediction - target) ** 2, axis=-1)
    
    if training:
      return prediction, target, error
    else:
      return error


class MLPNetwork(nj.Module):
  """Simple MLP for RND target/predictor networks."""
  
  layers: int = 3
  units: int = 512
  output_dim: int = 128
  act: str = 'silu'
  norm: str = 'rms'
  
  def __init__(self, **kw):
    self.kw = kw
  
  def __call__(self, x):
    x = nn.cast(x)
    
    # Hidden layers
    for i in range(self.layers):
      x = self.sub(f'layer{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'norm{i}', nn.Norm, self.norm)(x))
    
    # Output layer
    x = self.sub('out', nn.Linear, self.output_dim, **self.kw)(x)
    
    return x


def compute_rnd_reward_for_imagination(rnd_net, imgfeat):
  """
  Compute RND intrinsic rewards for imagined trajectories.
  
  Args:
    rnd_net: RNDNetwork module
    imgfeat: Imagined features dict with shape [B*K, H+1, ...]
  
  Returns:
    RND intrinsic rewards with shape [B*K, H+1]
  """
  BK, H1 = imgfeat['deter'].shape[:2]  # H1 = H + 1
  
  # Compute RND reward for each timestep
  rnd_rewards = []
  for t in range(H1):
    feat_t = jax.tree.map(lambda x: x[:, t], imgfeat)
    reward_t = rnd_net(feat_t, training=False)  # [B*K]
    rnd_rewards.append(reward_t)
  
  rnd_rewards = jnp.stack(rnd_rewards, axis=1)  # [B*K, H+1]
  
  return rnd_rewards


def compute_rnd_loss(rnd_net, feat):
  """
  Compute RND training loss (MSE between predictor and target).
  
  Args:
    rnd_net: RNDNetwork module
    feat: Features dict with shape [B, T, ...] or [B, ...]
  
  Returns:
    loss: Mean squared error [scalar]
  """
  # Handle both [B, T, ...] and [B, ...] shapes
  if feat['deter'].ndim == 3:
    # [B, T, ...] → flatten to [B*T, ...]
    B, T = feat['deter'].shape[:2]
    feat_flat = jax.tree.map(lambda x: x.reshape(B * T, *x.shape[2:]), feat)
  else:
    feat_flat = feat
  
  prediction, target, error = rnd_net(feat_flat, training=True)
  
  # MSE loss
  loss = jnp.mean((prediction - target) ** 2)
  
  return loss


# ============================================================================
# Symmetry-Aware Intrinsic Reward with Action Smoothness (Eq. 7)
# ============================================================================

def compute_symmetry_reward_for_imagination(
    hamiltonian_fn, imgfeat, imgact, lambda_smoothness=0.01, 
    use_ema_target=False, hamiltonian_fn_target=None):
  """
  Compute symmetry-aware intrinsic rewards for imagined trajectories (Eq. 7).
  
  Formula: r_sym = |H_target(Z_{t+1}) - H_target(Z_t)| - λ_s ||a_t - a_{t-1}||^2
  
  Physical Interpretation:
  - First term: Work done on the system (energy change due to actions)
  - Second term: Penalizes high-frequency action changes, encourages smooth exploration
  
  Using EMA target Hamiltonian (H_target) instead of current H provides more stable
  intrinsic rewards during early training when H is changing rapidly.
  
  Args:
    hamiltonian_fn: Function that computes H(feat, action) → scalar energy
    imgfeat: Imagined features dict with shape [B*K, H+1, ...]
    imgact: Imagined actions with shape [B*K, H+1, ...]
    lambda_smoothness: Weight for action smoothness penalty
    use_ema_target: If True, use EMA target Hamiltonian (more stable)
    hamiltonian_fn_target: EMA target Hamiltonian function (if use_ema_target=True)
  
  Returns:
    Symmetry intrinsic rewards with shape [B*K, H+1]
  """
  BK, H1 = imgfeat['deter'].shape[:2]  # H1 = H + 1
  
  # Select which Hamiltonian to use for computing energy
  H_fn = hamiltonian_fn_target if (use_ema_target and hamiltonian_fn_target is not None) else hamiltonian_fn
  
  # Compute energy H(z_t, a_t) for each timestep
  energies = []
  for t in range(H1):
    feat_t = jax.tree.map(lambda x: x[:, t], imgfeat)
    action_t = jax.tree.map(lambda x: x[:, t], imgact)
    energy_t = H_fn(feat_t, action_t)  # [B*K]
    energies.append(energy_t)
  
  energies = jnp.stack(energies, axis=1)  # [B*K, H+1]
  
  # Compute energy change: |H(z_{t+1}) - H(z_t)|
  energy_changes = jnp.abs(energies[:, 1:] - energies[:, :-1])  # [B*K, H]
  
  # Compute action smoothness penalty: ||a_t - a_{t-1}||^2
  # Extract action arrays (handle dict actions)
  if isinstance(imgact, dict):
    # Concatenate all action components
    act_components = []
    for key in sorted(imgact.keys()):
      act_val = imgact[key]
      if act_val.ndim > 2:
        act_val = act_val.reshape(BK, H1, -1)
      act_components.append(act_val)
    actions = jnp.concatenate(act_components, axis=-1)  # [B*K, H+1, act_dim]
  else:
    actions = imgact
    if actions.ndim == 2:
      actions = actions[:, :, None]  # Add feature dim if needed
  
  # Compute action differences
  action_diffs = actions[:, 1:] - actions[:, :-1]  # [B*K, H, act_dim]
  action_smoothness_penalty = jnp.sum(action_diffs ** 2, axis=-1)  # [B*K, H]
  
  # Combined symmetry reward (Eq. 7)
  sym_rewards = energy_changes - lambda_smoothness * action_smoothness_penalty  # [B*K, H]
  
  # Pad first timestep with zeros (no previous state to compare)
  sym_rewards = jnp.concatenate([jnp.zeros((BK, 1)), sym_rewards], axis=1)  # [B*K, H+1]
  
  return sym_rewards


def compute_action_smoothness_penalty(actions):
  """
  Compute action smoothness penalty: ||a_t - a_{t-1}||^2
  
  Args:
    actions: Action sequence with shape [B, T, act_dim]
  
  Returns:
    penalties: Smoothness penalties with shape [B, T-1]
  """
  action_diffs = actions[:, 1:] - actions[:, :-1]  # [B, T-1, act_dim]
  penalties = jnp.sum(action_diffs ** 2, axis=-1)  # [B, T-1]
  
  return penalties


# ============================================================================
# Hybrid Reward: RND + Symmetry with Annealing (Eq. 8)
# ============================================================================

def compute_hybrid_reward(
    rnd_reward, symmetry_reward, training_step, 
    anneal_start=0, anneal_steps=100000, w_final=1.0):
  """
  Compute hybrid intrinsic reward with annealing from RND to Symmetry (Eq. 8).
  
  Formula: r_int = (1 - w_t) * r_RND + w_t * r_sym
  
  Where w_t linearly anneals from 0 to w_final over anneal_steps.
  
  Motivation:
  - Early training: RND provides stable, general exploration
  - Late training: Symmetry reward provides physics-informed exploration
  - Annealing prevents instability from untrained Hamiltonian
  
  Args:
    rnd_reward: RND intrinsic reward [B, T]
    symmetry_reward: Symmetry intrinsic reward [B, T]
    training_step: Current training step (scalar)
    anneal_start: Step to start annealing
    anneal_steps: Number of steps to anneal from RND to Symmetry
    w_final: Final weight for symmetry reward (typically 1.0)
  
  Returns:
    hybrid_reward: Combined intrinsic reward [B, T]
    weight_sym: Current weight for symmetry reward (for logging)
  """
  # Compute annealing weight: w_t ∈ [0, w_final]
  progress = jnp.maximum(0.0, training_step - anneal_start) / anneal_steps
  weight_sym = jnp.minimum(progress, 1.0) * w_final
  weight_rnd = 1.0 - weight_sym
  
  # Hybrid reward (Eq. 8)
  hybrid_reward = weight_rnd * rnd_reward + weight_sym * symmetry_reward
  
  return hybrid_reward, weight_sym


def compute_hybrid_reward_for_imagination(
    rnd_net, hamiltonian_fn, imgfeat, imgact, training_step,
    lambda_smoothness=0.01, anneal_start=0, anneal_steps=100000, w_final=1.0,
    use_ema_target=False, hamiltonian_fn_target=None):
  """
  Compute hybrid intrinsic reward for imagined trajectories (combines RND + Symmetry).
  
  This is the main function to use during imagination-based policy learning.
  
  Args:
    rnd_net: RNDNetwork module
    hamiltonian_fn: Function that computes H(feat, action)
    imgfeat: Imagined features dict with shape [B*K, H+1, ...]
    imgact: Imagined actions with shape [B*K, H+1, ...]
    training_step: Current training step (for annealing)
    lambda_smoothness: Weight for action smoothness penalty
    anneal_start: Step to start annealing
    anneal_steps: Number of steps to anneal
    w_final: Final weight for symmetry reward
    use_ema_target: If True, use EMA target Hamiltonian
    hamiltonian_fn_target: EMA target Hamiltonian function
  
  Returns:
    hybrid_reward: Combined intrinsic reward [B*K, H+1]
    metrics: Dict with 'rnd_reward', 'sym_reward', 'weight_sym' for logging
  """
  # Compute both reward components
  rnd_reward = compute_rnd_reward_for_imagination(rnd_net, imgfeat)
  sym_reward = compute_symmetry_reward_for_imagination(
      hamiltonian_fn, imgfeat, imgact, lambda_smoothness,
      use_ema_target, hamiltonian_fn_target)
  
  # Combine with annealing
  hybrid_reward, weight_sym = compute_hybrid_reward(
      rnd_reward, sym_reward, training_step, 
      anneal_start, anneal_steps, w_final)
  
  # Metrics for logging
  metrics = {
      'rnd_reward': jnp.mean(rnd_reward),
      'sym_reward': jnp.mean(sym_reward),
      'weight_sym': weight_sym,
      'hybrid_reward': jnp.mean(hybrid_reward),
  }
  
  return hybrid_reward, metrics


# ============================================================================
# EMA Target Hamiltonian for Stable Intrinsic Rewards
# ============================================================================

class EMAHamiltonian(nj.Module):
  """
  Exponential Moving Average (EMA) target network for Hamiltonian.
  
  This provides a slowly-updating target for computing intrinsic rewards,
  which stabilizes training by preventing the reward signal from changing
  too rapidly as the Hamiltonian network is being trained.
  
  Similar to target networks in DQN and other RL algorithms.
  """
  
  ema_decay: float = 0.99
  
  def __init__(self, source_model, **kw):
    """
    Args:
      source_model: The main Hamiltonian network to track
      ema_decay: EMA decay rate (0.99 = 99% old, 1% new)
    """
    self.kw = kw
    self.source = source_model
    self._target_params = None
  
  def __call__(self, *args, **kwargs):
    """
    Compute Hamiltonian using EMA parameters.
    This uses the target parameters instead of the current ones.
    """
    # First call: initialize target params
    if self._target_params is None:
      # Run source model once to initialize parameters
      _ = self.source(*args, **kwargs)
      # Copy initial parameters
      self._target_params = jax.tree.map(lambda x: x, self.source.variables)
    
    # Temporarily swap to target parameters
    original_params = self.source.variables
    self.source.variables = self._target_params
    
    # Compute with target parameters
    output = self.source(*args, **kwargs)
    
    # Restore original parameters
    self.source.variables = original_params
    
    return output
  
  def update(self):
    """
    Update target parameters using EMA:
      θ_target = decay * θ_target + (1 - decay) * θ_source
    """
    if self._target_params is None:
      # Initialize on first update
      self._target_params = jax.tree.map(lambda x: x, self.source.variables)
    else:
      # EMA update
      def ema_update(target, source):
        return self.ema_decay * target + (1 - self.ema_decay) * source
      
      self._target_params = jax.tree.map(
          ema_update, self._target_params, self.source.variables)


def create_ema_hamiltonian(hamiltonian_rssm, ema_decay=0.99):
  """
  Create an EMA target Hamiltonian network.
  
  Args:
    hamiltonian_rssm: The HamiltonianRSSM module
    ema_decay: EMA decay rate
  
  Returns:
    A function that computes H using EMA parameters
  """
  # Create EMA wrapper
  ema_ham = EMAHamiltonian(hamiltonian_rssm, ema_decay=ema_decay)
  
  # Return a function that computes energy with EMA parameters
  def hamiltonian_fn_target(feat, action):
    return ema_ham.source.compute_hamiltonian(feat, action)
  
  return hamiltonian_fn_target, ema_ham


# ============================================================================
# ICM (Intrinsic Curiosity Module) for State-Prediction Based Exploration
# ============================================================================

class ICMNetwork(nj.Module):
  """
  Intrinsic Curiosity Module (ICM) for computing prediction-based intrinsic rewards.
  
  ICM consists of two networks:
  1. Forward model: Predicts next state feature φ(s_{t+1}) from φ(s_t) and a_t
  2. Inverse model: Predicts action â_t from φ(s_t) and φ(s_{t+1})
  
  The forward model prediction error serves as intrinsic reward:
    r_int = η * ||φ(s_{t+1}) - f(φ(s_t), a_t)||^2
  
  The inverse model serves two purposes:
  1. Learns a good feature representation φ that is sensitive to agent's actions
  2. Provides auxiliary supervision signal
  
  Reference: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", ICML 2017
  """
  
  layers: int = 3
  units: int = 512
  act: str = 'silu'
  norm: str = 'rms'
  feature_dim: int = 256
  forward_scale: float = 1.0
  inverse_scale: float = 0.1
  
  def __init__(self, action_dim, **kw):
    self.kw = kw
    self.action_dim = action_dim
    self._feature_net = None
    self._forward_model = None
    self._inverse_model = None
  
  def encode_state(self, feat):
    """
    Encode raw state features into ICM feature space φ(s).
    
    Args:
      feat: State features (dict with 'deter', 'stoch', etc.)
    
    Returns:
      Encoded features with shape [B, feature_dim]
    """
    # Concatenate all feature components
    if isinstance(feat, dict):
      feat_list = []
      if 'deter' in feat:
        feat_list.append(feat['deter'])
      if 'stoch' in feat:
        feat_list.append(feat['stoch'].reshape(feat['stoch'].shape[0], -1))
      elif 'q' in feat and 'p' in feat:
        q = feat['q'].reshape(feat['q'].shape[0], -1)
        p = feat['p'].reshape(feat['p'].shape[0], -1)
        feat_list.append(jnp.concatenate([q, p], axis=-1))
      x = jnp.concatenate(feat_list, axis=-1)
    else:
      x = feat
    
    x = nn.cast(x)
    
    # Lazy initialization
    if self._feature_net is None:
      self._feature_net = self.sub('feature', MLPNetwork,
                                   layers=self.layers,
                                   units=self.units,
                                   output_dim=self.feature_dim,
                                   act=self.act,
                                   norm=self.norm,
                                   **self.kw)
    
    return self._feature_net(x)
  
  def forward_model(self, phi_t, action_t):
    """
    Forward model: Predict next state feature φ_{t+1} from φ_t and a_t.
    
    Args:
      phi_t: Current state feature [B, feature_dim]
      action_t: Action [B, action_dim]
    
    Returns:
      Predicted next state feature [B, feature_dim]
    """
    # Concatenate state feature and action
    x = jnp.concatenate([phi_t, action_t], axis=-1)
    x = nn.cast(x)
    
    # Lazy initialization
    if self._forward_model is None:
      self._forward_model = self.sub('forward', MLPNetwork,
                                     layers=self.layers,
                                     units=self.units,
                                     output_dim=self.feature_dim,
                                     act=self.act,
                                     norm=self.norm,
                                     **self.kw)
    
    return self._forward_model(x)
  
  def inverse_model(self, phi_t, phi_t1):
    """
    Inverse model: Predict action â_t from φ_t and φ_{t+1}.
    
    Args:
      phi_t: Current state feature [B, feature_dim]
      phi_t1: Next state feature [B, feature_dim]
    
    Returns:
      Predicted action [B, action_dim]
    """
    # Concatenate current and next state features
    x = jnp.concatenate([phi_t, phi_t1], axis=-1)
    x = nn.cast(x)
    
    # Lazy initialization
    if self._inverse_model is None:
      self._inverse_model = self.sub('inverse', MLPNetwork,
                                     layers=self.layers,
                                     units=self.units,
                                     output_dim=self.action_dim,
                                     act=self.act,
                                     norm=self.norm,
                                     **self.kw)
    
    return self._inverse_model(x)
  
  def __call__(self, feat_t, feat_t1, action_t, training=False):
    """
    Compute ICM intrinsic reward and losses.
    
    Args:
      feat_t: Current state features [B, ...]
      feat_t1: Next state features [B, ...]
      action_t: Action taken [B, action_dim]
      training: If True, return losses for training
    
    Returns:
      If training=False: forward prediction error (intrinsic reward) [B]
      If training=True: (forward_loss, inverse_loss, intrinsic_reward)
    """
    # Encode states into feature space
    phi_t = self.encode_state(feat_t)
    phi_t1 = self.encode_state(feat_t1)
    
    # Forward model: predict next state feature
    phi_t1_pred = self.forward_model(phi_t, action_t)
    
    # Forward prediction error = intrinsic reward
    # Use stop gradient on phi_t1 to prevent feature collapse
    forward_error = jnp.mean(
        (phi_t1_pred - jax.lax.stop_gradient(phi_t1)) ** 2, axis=-1)
    
    intrinsic_reward = self.forward_scale * forward_error
    
    if not training:
      return intrinsic_reward
    
    # Inverse model: predict action
    action_pred = self.inverse_model(phi_t, phi_t1)
    
    # Inverse prediction loss
    inverse_loss = jnp.mean((action_pred - action_t) ** 2)
    
    # Forward loss (without stop gradient for training)
    forward_loss = jnp.mean((phi_t1_pred - phi_t1) ** 2)
    
    return forward_loss, inverse_loss, intrinsic_reward


def compute_icm_loss(icm_net, feat_seq, action_seq):
  """
  Compute ICM training losses (forward + inverse).
  
  Args:
    icm_net: ICMNetwork module
    feat_seq: Feature sequence dict with shape [B, T, ...]
    action_seq: Action sequence with shape [B, T, action_dim]
  
  Returns:
    total_loss: Combined forward and inverse loss [scalar]
    metrics: Dict with 'forward_loss' and 'inverse_loss'
  """
  B, T = feat_seq['deter'].shape[:2]
  
  # Extract consecutive state pairs
  feat_t = jax.tree.map(lambda x: x[:, :-1].reshape(B * (T-1), *x.shape[2:]), feat_seq)
  feat_t1 = jax.tree.map(lambda x: x[:, 1:].reshape(B * (T-1), *x.shape[2:]), feat_seq)
  action_t = action_seq[:, :-1].reshape(B * (T-1), -1)
  
  # Compute losses
  forward_loss, inverse_loss, _ = icm_net(feat_t, feat_t1, action_t, training=True)
  
  # Total loss = forward + inverse (with inverse scale)
  total_loss = forward_loss + icm_net.inverse_scale * inverse_loss
  
  metrics = {
      'icm_forward_loss': forward_loss,
      'icm_inverse_loss': inverse_loss,
  }
  
  return total_loss, metrics


def compute_icm_reward_for_imagination(icm_net, imgfeat, imgact):
  """
  Compute ICM intrinsic rewards for imagined trajectories.
  
  Args:
    icm_net: ICMNetwork module
    imgfeat: Imagined features dict with shape [B*K, H+1, ...]
    imgact: Imagined actions with shape [B*K, H+1, action_dim]
  
  Returns:
    ICM intrinsic rewards with shape [B*K, H+1]
  """
  BK, H1 = imgfeat['deter'].shape[:2]  # H1 = H + 1
  
  # Compute ICM reward for each transition
  icm_rewards = []
  
  # First timestep has no intrinsic reward (no previous state)
  icm_rewards.append(jnp.zeros(BK))
  
  for t in range(H1 - 1):
    feat_t = jax.tree.map(lambda x: x[:, t], imgfeat)
    feat_t1 = jax.tree.map(lambda x: x[:, t+1], imgfeat)
    action_t = imgact[:, t]
    
    reward_t = icm_net(feat_t, feat_t1, action_t, training=False)  # [B*K]
    icm_rewards.append(reward_t)
  
  icm_rewards = jnp.stack(icm_rewards, axis=1)  # [B*K, H+1]
  
  return icm_rewards


# ============================================================================
# Plan2Explore: Ensemble Disagreement for Exploration
# ============================================================================

class Plan2ExploreNetwork(nj.Module):
  """
  Plan2Explore: Exploration via ensemble disagreement.
  
  Uses an ensemble of forward dynamics models to predict next state features.
  The disagreement (variance) among ensemble predictions serves as intrinsic reward,
  indicating epistemic uncertainty about the dynamics.
  
  Key differences from ICM:
  1. Uses multiple models (ensemble) instead of single forward model
  2. Measures uncertainty via ensemble disagreement
  3. No inverse model needed
  
  Reference: Sekar et al., "Planning to Explore via Self-Supervised World Models", ICML 2020
  """
  
  num_ensemble: int = 5
  layers: int = 3
  units: int = 512
  act: str = 'silu'
  norm: str = 'rms'
  feature_dim: int = 256
  disagreement_scale: float = 1.0
  
  def __init__(self, action_dim, **kw):
    self.kw = kw
    self.action_dim = action_dim
    self._feature_net = None
    self._ensemble_models = None
  
  def encode_state(self, feat):
    """
    Encode raw state features into Plan2Explore feature space.
    
    Args:
      feat: State features (dict with 'deter', 'stoch', etc.)
    
    Returns:
      Encoded features with shape [B, feature_dim]
    """
    # Concatenate all feature components
    if isinstance(feat, dict):
      feat_list = []
      if 'deter' in feat:
        feat_list.append(feat['deter'])
      if 'stoch' in feat:
        feat_list.append(feat['stoch'].reshape(feat['stoch'].shape[0], -1))
      elif 'q' in feat and 'p' in feat:
        q = feat['q'].reshape(feat['q'].shape[0], -1)
        p = feat['p'].reshape(feat['p'].shape[0], -1)
        feat_list.append(jnp.concatenate([q, p], axis=-1))
      x = jnp.concatenate(feat_list, axis=-1)
    else:
      x = feat
    
    x = nn.cast(x)
    
    # Lazy initialization
    if self._feature_net is None:
      self._feature_net = self.sub('feature', MLPNetwork,
                                   layers=self.layers,
                                   units=self.units,
                                   output_dim=self.feature_dim,
                                   act=self.act,
                                   norm=self.norm,
                                   **self.kw)
    
    return self._feature_net(x)
  
  def ensemble_forward(self, phi_t, action_t):
    """
    Ensemble forward models: Each model predicts next state feature.
    
    Args:
      phi_t: Current state feature [B, feature_dim]
      action_t: Action [B, action_dim]
    
    Returns:
      List of predictions from each ensemble member [num_ensemble, B, feature_dim]
    """
    # Concatenate state feature and action
    x = jnp.concatenate([phi_t, action_t], axis=-1)
    x = nn.cast(x)
    
    # Lazy initialization: create ensemble of models
    if self._ensemble_models is None:
      self._ensemble_models = []
      for i in range(self.num_ensemble):
        model = self.sub(f'ensemble_{i}', MLPNetwork,
                        layers=self.layers,
                        units=self.units,
                        output_dim=self.feature_dim,
                        act=self.act,
                        norm=self.norm,
                        **self.kw)
        self._ensemble_models.append(model)
    
    # Get predictions from all ensemble members
    predictions = []
    for model in self._ensemble_models:
      pred = model(x)
      predictions.append(pred)
    
    return jnp.stack(predictions, axis=0)  # [num_ensemble, B, feature_dim]
  
  def __call__(self, feat_t, feat_t1, action_t, training=False):
    """
    Compute Plan2Explore intrinsic reward (ensemble disagreement).
    
    Args:
      feat_t: Current state features [B, ...]
      feat_t1: Next state features [B, ...]
      action_t: Action taken [B, action_dim]
      training: If True, return loss for training
    
    Returns:
      If training=False: ensemble disagreement (intrinsic reward) [B]
      If training=True: (prediction_loss, intrinsic_reward)
    """
    # Encode states into feature space
    phi_t = self.encode_state(feat_t)
    phi_t1 = self.encode_state(feat_t1)
    
    # Get ensemble predictions
    phi_t1_preds = self.ensemble_forward(phi_t, action_t)  # [num_ensemble, B, feature_dim]
    
    # Compute ensemble disagreement (variance across ensemble)
    # disagreement = Var[predictions] = E[pred^2] - E[pred]^2
    pred_mean = jnp.mean(phi_t1_preds, axis=0)  # [B, feature_dim]
    pred_var = jnp.var(phi_t1_preds, axis=0)  # [B, feature_dim]
    
    # Sum variance across feature dimensions for intrinsic reward
    disagreement = jnp.sum(pred_var, axis=-1)  # [B]
    intrinsic_reward = self.disagreement_scale * disagreement
    
    if not training:
      return intrinsic_reward
    
    # Training: minimize prediction error for all ensemble members
    # Use stop gradient on target to prevent feature collapse
    phi_t1_target = jax.lax.stop_gradient(phi_t1)
    
    # Compute MSE loss for each ensemble member
    prediction_errors = []
    for i in range(self.num_ensemble):
      pred = phi_t1_preds[i]  # [B, feature_dim]
      error = jnp.mean((pred - phi_t1_target) ** 2)
      prediction_errors.append(error)
    
    # Average loss across ensemble
    prediction_loss = jnp.mean(jnp.stack(prediction_errors))
    
    return prediction_loss, intrinsic_reward


def compute_plan2explore_loss(p2e_net, feat_seq, action_seq):
  """
  Compute Plan2Explore training loss (ensemble prediction error).
  
  Args:
    p2e_net: Plan2ExploreNetwork module
    feat_seq: Feature sequence dict with shape [B, T, ...]
    action_seq: Action sequence with shape [B, T, action_dim]
  
  Returns:
    loss: Ensemble prediction loss [scalar]
    metrics: Dict with 'p2e_loss' and 'disagreement'
  """
  B, T = feat_seq['deter'].shape[:2]
  
  # Extract consecutive state pairs
  feat_t = jax.tree.map(lambda x: x[:, :-1].reshape(B * (T-1), *x.shape[2:]), feat_seq)
  feat_t1 = jax.tree.map(lambda x: x[:, 1:].reshape(B * (T-1), *x.shape[2:]), feat_seq)
  action_t = action_seq[:, :-1].reshape(B * (T-1), -1)
  
  # Compute loss and disagreement
  loss, disagreement = p2e_net(feat_t, feat_t1, action_t, training=True)
  
  metrics = {
      'p2e_loss': loss,
      'p2e_disagreement': jnp.mean(disagreement),
  }
  
  return loss, metrics


def compute_plan2explore_reward_for_imagination(p2e_net, imgfeat, imgact):
  """
  Compute Plan2Explore intrinsic rewards for imagined trajectories.
  
  Args:
    p2e_net: Plan2ExploreNetwork module
    imgfeat: Imagined features dict with shape [B*K, H+1, ...]
    imgact: Imagined actions with shape [B*K, H+1, action_dim]
  
  Returns:
    Plan2Explore intrinsic rewards with shape [B*K, H+1]
  """
  BK, H1 = imgfeat['deter'].shape[:2]  # H1 = H + 1
  
  # Compute disagreement reward for each transition
  p2e_rewards = []
  
  # First timestep has no intrinsic reward (no previous state)
  p2e_rewards.append(jnp.zeros(BK))
  
  for t in range(H1 - 1):
    feat_t = jax.tree.map(lambda x: x[:, t], imgfeat)
    feat_t1 = jax.tree.map(lambda x: x[:, t+1], imgfeat)
    action_t = imgact[:, t]
    
    reward_t = p2e_net(feat_t, feat_t1, action_t, training=False)  # [B*K]
    p2e_rewards.append(reward_t)
  
  p2e_rewards = jnp.stack(p2e_rewards, axis=1)  # [B*K, H+1]
  
  return p2e_rewards
