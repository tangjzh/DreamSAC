import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm
from . import hamiltonian_rssm
from . import intrinsic
from . import savi_encoder
from . import savi_decoder
from . import viewpoint

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                     ___   _   ___ ---",
      r"--- |   \ _ _ ___ __ _ _ __ / __| /_\ / __|---",
      r"--- | |) | '_/ -_) _` | '  \\__ \/ _ \ (__ ---",
      r"--- |___/|_| \___\__,_|_|_|_|___/_/ \_\___|---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    
    # Different encoders have different __init__ signatures
    if config.enc.typ == 'simple':
      # simple encoder: Encoder(obs_space, **kw)
      self.enc = rssm.Encoder(enc_space, **config.enc[config.enc.typ], name='enc')
    elif config.enc.typ == 'savi':
      # SAVi encoder: SAViEncoder(obs_space, act_space, **kw)
      self.enc = savi_encoder.SAViEncoder(enc_space, act_space, **config.enc[config.enc.typ], name='enc')
    else:
      raise ValueError(f"Unknown encoder type: {config.enc.typ}")
    self.dyn = {
        'rssm': rssm.RSSM,
        'hamiltonian': hamiltonian_rssm.HamiltonianRSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
        'savi': savi_decoder.SAViDecoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    # Handle both RSSM (with 'stoch') and HamiltonianRSSM (with 'q' and 'p')
    if config.dyn.typ == 'hamiltonian':
      self.feat2tensor = lambda x: jnp.concatenate([
          nn.cast(x['deter']),
          nn.cast(x['q'].reshape((*x['q'].shape[:-2], -1))),
          nn.cast(x['p'].reshape((*x['p'].shape[:-2], -1)))], -1)
    else:
      self.feat2tensor = lambda x: jnp.concatenate([
          nn.cast(x['deter']),
          nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # Initialize RND network for novelty-based exploration (if enabled)
    self.rnd_net = None
    if config.use_rnd:
      self.rnd_net = intrinsic.RNDNetwork(**config.rnd, name='rnd')
    
    # Initialize ICM network for curiosity-driven exploration (if enabled)
    self.icm_net = None
    if config.use_icm:
      # Get action dimension (convert to Python int)
      action_dim = int(sum([np.prod(v.shape) for v in act_space.values()]))
      self.icm_net = intrinsic.ICMNetwork(action_dim, **config.icm, name='icm')
    
    # Initialize Plan2Explore network for ensemble-based exploration (if enabled)
    self.p2e_net = None
    if config.use_plan2explore:
      # Get action dimension (convert to Python int)
      action_dim = int(sum([np.prod(v.shape) for v in act_space.values()]))
      self.p2e_net = intrinsic.Plan2ExploreNetwork(action_dim, **config.plan2explore, name='p2e')
    
    # Training step counter for annealing (used in hybrid intrinsic reward)
    self.training_step = nj.Variable(jnp.array, 0, i32, name='training_step')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    # Add intrinsic reward networks to modules if enabled
    if self.rnd_net is not None:
      self.modules.append(self.rnd_net)
    if self.icm_net is not None:
      self.modules.append(self.icm_net)
    if self.p2e_net is not None:
      self.modules.append(self.p2e_net)
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    # Remove vr loss scale if viewpoint robustness is not used
    if not self.config.use_viewpoint_robustness:
      scales.pop('vr', None)
    # Remove rnd loss scale if RND is not used
    if not self.config.use_rnd:
      scales.pop('rnd', None)
    # Remove icm loss scale if ICM is not used
    if not self.config.use_icm:
      scales.pop('icm', None)
    # Remove p2e loss scale if Plan2Explore is not used
    if not self.config.use_plan2explore:
      scales.pop('p2e', None)
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    
    # Use random actions if random_policy is enabled
    if self.config.random_policy:
      batch_size = reset.shape[0]
      # Sample random actions from action space using JAX random
      act = {}
      for k, v in self.act_space.items():
        if k == 'reset':
          continue
        shape = (batch_size,) + v.shape
        if v.discrete:
          # Discrete action: uniform random from [v.low, v.high)
          act[k] = jax.random.randint(nj.seed(), shape, v.low, v.high, dtype=v.dtype)
        else:
          # Continuous action: uniform random from [v.low, v.high]
          act[k] = jax.random.uniform(nj.seed(), shape, minval=v.low, maxval=v.high, dtype=v.dtype)
      act = jax.tree.map(nn.cast, act)
    else:
      policy = self.pol(self.feat2tensor(feat), bdims=1)
      act = sample(policy)
    
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    # Increment training step counter for intrinsic reward annealing
    self.training_step.write(self.training_step.read() + 1)
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    initial_enc_carry = enc_carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)

    # Viewpoint Robustness Loss (L_vr)
    if self.config.use_viewpoint_robustness:
      if training:
        # Extract images from observations
        imgkeys = [k for k, v in self.obs_space.items() if isimage(v)]
        if imgkeys:
          images = jnp.concatenate([obs[k] for k in sorted(imgkeys)], -1)
          # Normalize to [0, 1]
          images = f32(images) / 255.0
          
          # Flatten batch and time dimensions for augmentation
          images_flat = images.reshape((B * T, *images.shape[2:]))
          
          # Create encoder function that takes images -> latent
          def encode_fn(imgs):
            augmented = dict(obs)
            offset = 0
            for key in sorted(imgkeys):
              channels = obs[key].shape[-1]
              values = imgs[..., offset:offset + channels]
              augmented[key] = jnp.clip(values * 255, 0, 255).astype(
                  jnp.uint8).reshape(obs[key].shape)
              offset += channels
            _, _, view_tokens = self.enc(
                initial_enc_carry, augmented, reset, training)
            return view_tokens.reshape((B * T, -1))
          
          # Compute L_vr using the flat images
          vr_loss, vr_metrics = viewpoint.compute_viewpoint_robustness_loss(
              encode_fn, images_flat, 
              key=nj.seed(), 
              temperature=self.config.viewpoint_temperature)
          
          # Add to losses - broadcast scalar to [B, T] for consistency
          losses['vr'] = vr_loss * jnp.ones((B, T))
          metrics.update(prefix(vr_metrics, 'vr'))
        else:
          # No image observations, add zero loss
          losses['vr'] = jnp.zeros((B, T))
      else:
        # Not training, add zero loss for consistency with self.scales
        losses['vr'] = jnp.zeros((B, T))
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # RND Loss: Train predictor network to match fixed target network
    if self.rnd_net is not None:
      rnd_loss = intrinsic.compute_rnd_loss(self.rnd_net, repfeat)
      losses['rnd'] = rnd_loss * jnp.ones((B, T))  # Broadcast to [B, T] for consistency
      if training:
        metrics['train/rnd_loss'] = rnd_loss
    
    # ICM Loss: Train forward and inverse models
    if self.icm_net is not None:
      # Prepare actions for ICM (concatenate all action components)
      action_components = [prevact[k] for k in sorted(prevact.keys())]
      action_seq = jnp.concatenate([a.reshape(B, T, -1) for a in action_components], axis=-1)
      
      icm_loss, icm_metrics = intrinsic.compute_icm_loss(self.icm_net, repfeat, action_seq)
      losses['icm'] = icm_loss * jnp.ones((B, T))  # Broadcast to [B, T] for consistency
      if training:
        metrics['train/icm_forward_loss'] = icm_metrics['icm_forward_loss']
        metrics['train/icm_inverse_loss'] = icm_metrics['icm_inverse_loss']
    
    # Plan2Explore Loss: Train ensemble of forward models
    if self.p2e_net is not None:
      # Prepare actions for Plan2Explore (concatenate all action components)
      action_components = [prevact[k] for k in sorted(prevact.keys())]
      action_seq = jnp.concatenate([a.reshape(B, T, -1) for a in action_components], axis=-1)
      
      p2e_loss, p2e_metrics = intrinsic.compute_plan2explore_loss(self.p2e_net, repfeat, action_seq)
      losses['p2e'] = p2e_loss * jnp.ones((B, T))  # Broadcast to [B, T] for consistency
      if training:
        metrics['train/p2e_loss'] = p2e_metrics['p2e_loss']
        metrics['train/p2e_disagreement'] = p2e_metrics['p2e_disagreement']

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    
    # Compute rewards (external or intrinsic)
    if self.config.use_intrinsic_reward:
      # Get current training step for annealing (used in hybrid reward)
      training_step = f32(self.training_step.read())
      
      # Compute intrinsic reward based on configured type
      if self.config.intrinsic_reward_type == 'rnd' and self.rnd_net is not None:
        # Pure RND exploration
        rew_pred = intrinsic.compute_rnd_reward_for_imagination(
            self.rnd_net, imgfeat)
        metrics['intrinsic_reward/rnd'] = rew_pred.mean()
        
      elif self.config.intrinsic_reward_type == 'symmetry' and self.config.dyn.typ == 'hamiltonian':
        # Pure symmetry-based exploration (original behavior)
        rew_pred = intrinsic.compute_symmetry_reward_for_imagination(
            lambda feat, act: self.dyn.compute_hamiltonian(feat, act),
            imgfeat, imgact, 
            lambda_smoothness=self.config.lambda_smoothness)
        metrics['intrinsic_reward/symmetry'] = rew_pred.mean()
        
      elif self.config.intrinsic_reward_type == 'hybrid' and self.rnd_net is not None and self.config.dyn.typ == 'hamiltonian':
        # Hybrid: RND + Symmetry with annealing
        rew_pred, hybrid_metrics = intrinsic.compute_hybrid_reward_for_imagination(
            self.rnd_net,
            lambda feat, act: self.dyn.compute_hamiltonian(feat, act),
            imgfeat, imgact, training_step,
            lambda_smoothness=self.config.lambda_smoothness,
            anneal_start=self.config.anneal_start,
            anneal_steps=self.config.anneal_steps,
            w_final=self.config.w_final)
        metrics.update({f'intrinsic_reward/{k}': v for k, v in hybrid_metrics.items()})
        
      elif self.config.intrinsic_reward_type == 'icm' and self.icm_net is not None:
        # ICM: Curiosity-driven exploration via forward model prediction error
        # Prepare actions for ICM (concatenate all action components)
        imgact_flat = jnp.concatenate([imgact[k].reshape(B * K, H + 1, -1) 
                                       for k in sorted(imgact.keys())], axis=-1)
        rew_pred = intrinsic.compute_icm_reward_for_imagination(
            self.icm_net, imgfeat, imgact_flat)
        metrics['intrinsic_reward/icm'] = rew_pred.mean()
        
      elif self.config.intrinsic_reward_type == 'plan2explore' and self.p2e_net is not None:
        # Plan2Explore: Exploration via ensemble disagreement
        # Prepare actions for Plan2Explore (concatenate all action components)
        imgact_flat = jnp.concatenate([imgact[k].reshape(B * K, H + 1, -1) 
                                       for k in sorted(imgact.keys())], axis=-1)
        rew_pred = intrinsic.compute_plan2explore_reward_for_imagination(
            self.p2e_net, imgfeat, imgact_flat)
        metrics['intrinsic_reward/plan2explore'] = rew_pred.mean()
        
      else:
        # Fallback: use external reward prediction
        rew_pred = self.rew(inp, 2).pred()
        
      # Scale intrinsic reward
      if self.config.intrinsic_reward_type in ['rnd', 'symmetry', 'hybrid', 'icm', 'plan2explore']:
        rew_pred = rew_pred * self.config.intrinsic_reward_scale
        metrics['intrinsic_reward/total'] = rew_pred.mean()
    else:
      # Use standard external reward prediction
      rew_pred = self.rew(inp, 2).pred()
    
    los, imgloss_out, mets = imag_loss(
        imgact,
        rew_pred,
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
