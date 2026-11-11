"""
SAVi (Slot Attention for Video) Encoder for DreamerV3.

Based on:
- Slot Attention: https://arxiv.org/pdf/2006.15055
- SAVi: https://arxiv.org/abs/2111.13010

Adapted to JAX from the PyTorch implementation in SOLD.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import embodied.jax.nets as nn
from typing import Optional

f32 = jnp.float32


class ImagePositionalEmbedding(nj.Module):
  """Soft image positional embedding for CNN features."""
  
  feature_dim: int = 64
  
  def __init__(self, image_size, **kw):
    self.image_size = image_size
    self.kw = kw
    
  def __call__(self, embeddings):
    """
    Args:
      embeddings: [B, H, W, C] feature map
    Returns:
      embeddings + positional encoding: [B, H, W, C]
    """
    batch_size, height, width, num_channels = embeddings.shape
    
    # Create coordinate grid [0, 1] in [x, y, -x, -y] directions
    y = jnp.linspace(-1.0, 1.0, height)
    x = jnp.linspace(-1.0, 1.0, width)
    yy, xx = jnp.meshgrid(y, x, indexing='ij')
    
    # Stack: [H, W, 4]
    grid = jnp.stack([yy, xx, 1.0 - yy, 1.0 - xx], axis=-1)
    
    # Expand batch dimension and project to feature dim
    grid = grid[None, :, :, :]  # [1, H, W, 4]
    grid = jnp.tile(grid, (batch_size, 1, 1, 1))
    
    # Convert grid to same dtype as embeddings (e.g., bfloat16)
    grid = grid.astype(embeddings.dtype)
    
    # Project 4-channel grid to match embeddings channel dimension
    projected = self.sub('pos_proj', nn.Linear, num_channels, **self.kw)(grid)
    
    return embeddings + projected


class SlotAttention(nj.Module):
  """Slot Attention mechanism (Corrector in SAVi terminology)."""
  
  num_slots: int = 7
  slot_dim: int = 128
  num_iterations: int = 3
  num_initial_iterations: int = 5
  hidden_dim: int = 128
  epsilon: float = 1e-8
  norm: str = 'layer'
  act: str = 'relu'
  
  def __init__(self, feature_dim, **kw):
    self.feature_dim = feature_dim
    self.scale = feature_dim ** -0.5
    self.kw = kw
    
  def __call__(self, features, slots, is_first=False):
    """
    Apply slot attention.
    
    Args:
      features: [B, N_locs, feature_dim] - flattened spatial features
      slots: [B, num_slots, slot_dim] - slot initializations/predictions
      is_first: bool - whether this is the first timestep (static argument)
    
    Returns:
      slots: [B, num_slots, slot_dim] - refined slots
    """
    batch_size = features.shape[0]
    
    # Normalize inputs
    features = self.sub('feat_norm', nn.Norm, 'layer')(features)
    
    # Project features to slot_dim
    k = self.sub('to_k', nn.Linear, self.slot_dim, **self.kw)(features)
    v = self.sub('to_v', nn.Linear, self.slot_dim, **self.kw)(features)
    
    # Iterative refinement
    # Use more iterations for initial frames (is_first=True) for better initialization
    # Note: We use max iterations and conditionally execute to keep JIT-friendly
    max_iters = max(self.num_iterations, self.num_initial_iterations)
    
    # Convert scale and epsilon to correct dtype once
    scale = nn.cast(jnp.array(self.scale))
    epsilon = nn.cast(jnp.array(self.epsilon))
    
    for i in range(max_iters):
      # Skip this iteration if we've exceeded the needed iterations
      # For initial frames: run num_initial_iterations
      # For other frames: run num_iterations
      should_run = jnp.logical_or(
          i < self.num_iterations,
          jnp.logical_and(is_first, i < self.num_initial_iterations)
      )
      
      # Only update if should_run (using jnp.where to stay JIT-compatible)
      # Store previous slots in case we skip
      slots_before_iter = slots
      slots_prev = slots
      
      # Normalize slots
      slots = self.sub(f'slot_norm_{i}', nn.Norm, 'layer')(slots)
      
      # Query projection
      q = self.sub('to_q', nn.Linear, self.slot_dim, **self.kw)(slots)
      
      # Attention: [B, num_slots, N_locs]
      dots = jnp.einsum('bid,bjd->bij', q, k) * scale
      attn = jax.nn.softmax(dots, axis=1) + epsilon
      attn = attn / jnp.sum(attn, axis=-1, keepdims=True)
      
      # Weighted mean: [B, num_slots, slot_dim]
      updates = jnp.einsum('bij,bjd->bid', attn, v)
      
      # GRU update
      slots_updated = self._gru_update(updates, slots_prev, i)
      
      # MLP
      slots_updated = slots_updated + self._mlp(slots_updated, i)
      
      # Conditionally apply update based on should_run
      # This allows us to use different iteration counts while staying JIT-friendly
      # should_run is a scalar boolean, so we expand it to match slots shape
      slots = jnp.where(jnp.asarray(should_run)[..., None, None],
                        slots_updated, slots_before_iter)
    
    return slots
  
  def _gru_update(self, updates, slots_prev, iteration):
    """
    GRU cell for slot update.
    Matches PyTorch nn.GRUCell implementation:
    r = sigmoid(W_ir @ x + W_hr @ h + b)
    z = sigmoid(W_iz @ x + W_hz @ h + b)  
    n = tanh(W_in @ x + r * (W_hn @ h + b))
    h' = (1 - z) * n + z * h
    
    Note: Parameters are shared across all iterations (not iteration-specific)
    """
    # Flatten batch dimensions for GRU computation
    batch_size = updates.shape[0]
    updates_flat = updates.reshape(-1, self.slot_dim)
    slots_prev_flat = slots_prev.reshape(-1, self.slot_dim)
    
    # Reset gate: r = sigmoid(W_ir @ updates + W_hr @ slots_prev)
    r_x = self.sub('gru_r_x', nn.Linear, self.slot_dim, **self.kw)(updates_flat)
    r_h = self.sub('gru_r_h', nn.Linear, self.slot_dim, **self.kw)(slots_prev_flat)
    r = jax.nn.sigmoid(r_x + r_h)
    
    # Update gate: z = sigmoid(W_iz @ updates + W_hz @ slots_prev)
    z_x = self.sub('gru_z_x', nn.Linear, self.slot_dim, **self.kw)(updates_flat)
    z_h = self.sub('gru_z_h', nn.Linear, self.slot_dim, **self.kw)(slots_prev_flat)
    z = jax.nn.sigmoid(z_x + z_h)
    
    # New gate: n = tanh(W_in @ updates + r * W_hn @ slots_prev)
    n_x = self.sub('gru_n_x', nn.Linear, self.slot_dim, **self.kw)(updates_flat)
    n_h = self.sub('gru_n_h', nn.Linear, self.slot_dim, **self.kw)(slots_prev_flat)
    n = jnp.tanh(n_x + r * n_h)
    
    # Final update: h' = (1 - z) * n + z * h
    slots_flat = (1 - z) * n + z * slots_prev_flat
    
    # Reshape back to original batch structure
    slots = slots_flat.reshape(batch_size, -1, self.slot_dim)
    return slots
  
  def _mlp(self, slots, iteration):
    """MLP for slot refinement."""
    slots_norm = self.sub(f'mlp_norm_{iteration}', nn.Norm, 'layer')(slots)
    x = self.sub(f'mlp1_{iteration}', nn.Linear, self.hidden_dim, **self.kw)(slots_norm)
    x = nn.act(self.act)(x)
    x = self.sub(f'mlp2_{iteration}', nn.Linear, self.slot_dim, **self.kw)(x)
    return x


class SlotInitializer(nj.Module):
  """Learned random slot initialization."""
  
  num_slots: int = 7
  slot_dim: int = 128
  
  def __init__(self, **kw):
    self.kw = kw
    
  def __call__(self, batch_size):
    """
    Sample initial slots from learned Gaussian.
    Uses Xavier uniform initialization matching SOLD implementation.
    
    Returns:
      slots: [B, num_slots, slot_dim]
    """
    # Learned mean and sigma parameters
    shape = (1, 1, self.slot_dim)
    
    # Xavier uniform initialization: limit = sqrt(6.0 / (1 + slot_dim))
    # This matches the SOLD implementation
    limit = jnp.sqrt(6.0 / (1 + self.slot_dim))
    
    # Initialize with uniform distribution in [-limit, limit]
    def uniform_init(shape):
      return jax.random.uniform(
          nj.seed(), shape, minval=-limit, maxval=limit, dtype=f32)
    
    # Create learnable parameters using self.value with proper initialization
    mu = self.value('slots_mu', uniform_init, shape)
    sigma = self.value('slots_sigma', uniform_init, shape)
    
    # Expand to batch and slot dimensions
    mu = jnp.tile(mu, (batch_size, self.num_slots, 1))
    sigma = jnp.tile(sigma, (batch_size, self.num_slots, 1))
    
    # Sample from N(mu, sigma)
    eps = jax.random.normal(nj.seed(), mu.shape)
    slots = mu + sigma * eps
    
    return nn.cast(slots)


class SlotPredictor(nj.Module):
  """Transformer-based slot predictor for temporal consistency."""
  
  slot_dim: int = 128
  num_heads: int = 4
  mlp_dim: int = 512
  norm: str = 'layer'
  act: str = 'relu'
  
  def __init__(self, action_dim, **kw):
    self.action_dim = action_dim
    self.kw = kw
    
  def __call__(self, slots, action):
    """
    Predict next slots given current slots and action.
    
    Args:
      slots: [B, num_slots, slot_dim]
      action: [B, action_dim]
    
    Returns:
      predicted_slots: [B, num_slots, slot_dim]
    """
    batch_size, num_slots, slot_dim = slots.shape
    
    # Embed action
    action_emb = self.sub('action_embed', nn.Linear, self.slot_dim, **self.kw)(action)
    action_token = action_emb[:, None, :]  # [B, 1, slot_dim]
    
    # Concatenate slots and action token
    tokens = jnp.concatenate([slots, action_token], axis=1)  # [B, num_slots+1, slot_dim]
    
    # Multi-head Self-attention (matching SOLD TransformerBlock)
    tokens_norm = self.sub('attn_norm', nn.Norm, 'layer')(tokens)
    
    # Project to Q, K, V
    q = self.sub('q_proj', nn.Linear, self.slot_dim, **self.kw)(tokens_norm)
    k = self.sub('k_proj', nn.Linear, self.slot_dim, **self.kw)(tokens_norm)
    v = self.sub('v_proj', nn.Linear, self.slot_dim, **self.kw)(tokens_norm)
    
    # Reshape for multi-head attention
    # [B, num_tokens, slot_dim] -> [B, num_tokens, num_heads, head_dim]
    num_tokens = tokens.shape[1]
    head_dim = self.slot_dim // self.num_heads
    assert self.slot_dim % self.num_heads == 0, \
        f"slot_dim ({self.slot_dim}) must be divisible by num_heads ({self.num_heads})"
    
    q = q.reshape(batch_size, num_tokens, self.num_heads, head_dim)
    k = k.reshape(batch_size, num_tokens, self.num_heads, head_dim)
    v = v.reshape(batch_size, num_tokens, self.num_heads, head_dim)
    
    # Transpose to [B, num_heads, num_tokens, head_dim]
    q = q.transpose((0, 2, 1, 3))
    k = k.transpose((0, 2, 1, 3))
    v = v.transpose((0, 2, 1, 3))
    
    # Compute attention for each head
    # [B, num_heads, num_tokens, head_dim] @ [B, num_heads, head_dim, num_tokens]
    # -> [B, num_heads, num_tokens, num_tokens]
    scale = head_dim ** -0.5
    dots = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
    attn = jax.nn.softmax(dots, axis=-1)
    
    # Apply attention to values
    # [B, num_heads, num_tokens, num_tokens] @ [B, num_heads, num_tokens, head_dim]
    # -> [B, num_heads, num_tokens, head_dim]
    out = jnp.einsum('bhij,bhjd->bhid', attn, v)
    
    # Transpose back and reshape
    # [B, num_heads, num_tokens, head_dim] -> [B, num_tokens, num_heads, head_dim]
    out = out.transpose((0, 2, 1, 3))
    # -> [B, num_tokens, slot_dim]
    out = out.reshape(batch_size, num_tokens, self.slot_dim)
    
    # Output projection (matching SOLD's out_projection)
    out = self.sub('out_proj', nn.Linear, self.slot_dim, **self.kw)(out)
    
    # Residual connection
    tokens = tokens + out
    
    # MLP
    tokens_norm = self.sub('mlp_norm', nn.Norm, 'layer')(tokens)
    mlp_out = self.sub('mlp1', nn.Linear, self.mlp_dim, **self.kw)(tokens_norm)
    mlp_out = nn.act(self.act)(mlp_out)
    mlp_out = self.sub('mlp2', nn.Linear, self.slot_dim, **self.kw)(mlp_out)
    tokens = tokens + mlp_out
    
    # Extract predicted slots (exclude action token)
    predicted_slots = tokens[:, :-1, :]
    
    return predicted_slots


class SAViEncoder(nj.Module):
  """
  SAVi Encoder: Object-centric encoder using slot attention.
  
  Replaces the standard CNN encoder in DreamerV3 with a slot-based encoder
  that decomposes scenes into object representations.
  """
  
  # CNN encoder params
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  kernel: int = 5
  
  # Slot attention params
  num_slots: int = 7
  slot_dim: int = 128
  feature_dim: int = 128
  num_iterations: int = 3
  num_initial_iterations: int = 5
  
  # Standard params
  act: str = 'silu'
  norm: str = 'rms'
  winit: str = 'trunc_normal_in'  # Weight initialization (for compatibility with config)
  
  def __init__(self, obs_space, act_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.act_space = act_space
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw
    
  @property
  def entry_space(self):
    import elements
    return {'prev_slots': elements.Space(
        np.float32, (self.num_slots, self.slot_dim))}
  
  def initial(self, batch_size):
    # Store previous slots for temporal consistency
    # Use nn.cast to ensure correct dtype (bfloat16 for computation)
    zeros = jnp.zeros((batch_size, self.num_slots, self.slot_dim), f32)
    return {'prev_slots': nn.cast(zeros)}
  
  def truncate(self, entries, carry=None):
    return {'prev_slots': nn.cast(entries['prev_slots'][:, -1])}
  
  def __call__(self, carry, obs, reset, training, single=False):
    """
    Encode observation to slots.
    
    Args:
      carry: dict with 'prev_slots'
      obs: dict with image observations
      reset: [B] or [B, T] boolean reset flags
      training: bool
      single: bool - if True, process single timestep
    
    Returns:
      carry: updated carry
      entry: empty dict (for compatibility)
      tokens: [B, (T,) num_slots * slot_dim] - flattened slots as tokens
    """
    bdims = 1 if single else 2
    bshape = reset.shape
    
    # Extract images
    imgs = [obs[k] for k in sorted(self.imgkeys)]
    assert all(x.dtype == jnp.uint8 for x in imgs)
    x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255.0 - 0.5
    x = x.reshape((-1, *x.shape[bdims:]))  # Flatten batch/time
    
    # CNN backbone to extract features
    K = self.kernel
    for i, depth in enumerate(self.depths):
      x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
      B, H, W, C = x.shape
      x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
      x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
    
    # x shape: [B*T, H', W', C']
    batch_time, h, w, c = x.shape
    
    # Add positional encoding
    x = self.sub('pos_embed', ImagePositionalEmbedding, (h, w), 
                 feature_dim=c, **self.kw)(x)
    
    # Project to feature_dim and flatten spatial dimensions
    features = x.reshape((batch_time, h * w, c))
    features = self.sub('feature_mlp', nn.Linear, self.feature_dim, **self.kw)(features)
    features = nn.act(self.act)(features)
    
    initializer = self.sub('slot_init', SlotInitializer,
                           num_slots=self.num_slots,
                           slot_dim=self.slot_dim, **self.kw)
    slot_attention = self.sub('slot_attn', SlotAttention,
                              feature_dim=self.feature_dim,
                              num_slots=self.num_slots,
                              slot_dim=self.slot_dim,
                              num_iterations=self.num_iterations,
                              num_initial_iterations=self.num_initial_iterations,
                              **self.kw)
    features = features.reshape((*bshape, h * w, self.feature_dim))

    def step(previous, inputs):
      feature, first = inputs
      initial = initializer(feature.shape[0])
      slots = jnp.where(first[:, None, None], initial, previous)
      slots = slot_attention(feature, nn.cast(slots), is_first=first)
      return slots, slots

    if single:
      last, slots = step(nn.cast(carry['prev_slots']), (features, reset))
    else:
      last, slots = nj.scan(step, nn.cast(carry['prev_slots']),
                            (features, reset), axis=1)
    tokens = slots.reshape((*bshape, self.num_slots * self.slot_dim))
    return {'prev_slots': last}, {'prev_slots': slots}, tokens
