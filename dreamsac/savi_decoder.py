"""
SAVi Decoder for object-centric reconstruction.

Decodes slots into individual object RGB + masks, then composites them.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import embodied.jax
import embodied.jax.nets as nn
from .savi_encoder import ImagePositionalEmbedding

f32 = jnp.float32


class SAViDecoder(nj.Module):
  """
  SAVi Decoder: Decode slots to RGB + mask per slot, then composite.
  """
  
  # Decoder CNN params
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  kernel: int = 5
  
  # Slot params
  num_slots: int = 7
  slot_dim: int = 128
  
  # Standard params
  act: str = 'silu'
  norm: str = 'rms'
  outscale: float = 1.0
  winit: str = 'trunc_normal_in'  # Weight initialization (for compatibility with config)
  
  def __init__(self, obs_space, **kw):
    self.obs_space = obs_space
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw
    
    # Infer image size from obs_space
    img_shapes = [obs_space[k].shape for k in self.imgkeys]
    self.image_size = img_shapes[0][:2]  # (H, W)
    self.num_channels = sum(s[2] for s in img_shapes)
    
  @property
  def entry_space(self):
    return {}
  
  def initial(self, batch_size):
    return {}
  
  def truncate(self, entries, carry=None):
    return {}
  
  def __call__(self, carry, feat, reset, training, single=False):
    """
    Decode features (slots) to reconstructed images.
    
    Args:
      carry: decoder carry (empty for now)
      feat: dict with 'slots' [B, (T,) num_slots, slot_dim]
      reset: [B] or [B, T] boolean
      training: bool
      single: bool
    
    Returns:
      carry: updated carry
      entry: empty dict
      recons: dict with reconstructed images per key
    """
    bdims = 1 if single else 2
    bshape = reset.shape
    
    # Extract slots from feat
    # feat is actually a dict from HamiltonianRSSM with keys like 'q', 'p', 'deter'
    # We need to extract slots from the feature representation
    # For now, assume feat has shape [B, (T,) feature_dim]
    # and we need to reshape it to slots
    
    if isinstance(feat, dict):
      # Concatenate all features
      feat_tensor = jnp.concatenate([
        feat['deter'],
        feat.get('q', jnp.array([])).reshape((*feat['deter'].shape[:-1], -1)),
        feat.get('p', jnp.array([])).reshape((*feat['deter'].shape[:-1], -1))
      ], axis=-1) if 'q' in feat else feat['deter']
    else:
      feat_tensor = feat
    
    # Reshape to extract slots
    # Assume the token dimension contains flattened slots
    feat_flat = feat_tensor.reshape((-1, feat_tensor.shape[-1]))
    
    # If feature_dim == num_slots * slot_dim, reshape directly
    if feat_tensor.shape[-1] == self.num_slots * self.slot_dim:
      slots = feat_flat.reshape((-1, self.num_slots, self.slot_dim))
    else:
      # Otherwise, use an MLP to project to slot space
      slots_flat = self.sub('feat_to_slots', nn.Linear, 
                           self.num_slots * self.slot_dim, **self.kw)(feat_flat)
      slots = slots_flat.reshape((-1, self.num_slots, self.slot_dim))
    
    # Decode each slot independently
    batch_time, num_slots, slot_dim = slots.shape
    slots_flat = slots.reshape((batch_time * num_slots, slot_dim))
    
    # Spatial broadcast: expand slots to image size
    h, w = self.image_size
    # Reduce spatial resolution for decoder (encoder downsampled by 2^len(depths))
    decoder_h = h // (2 ** len(self.depths))
    decoder_w = w // (2 ** len(self.depths))
    
    # Broadcast slot to spatial dimensions
    # slots_flat: [B*T*num_slots, slot_dim]
    x = slots_flat[:, None, None, :]  # [B*T*num_slots, 1, 1, slot_dim]
    x = jnp.tile(x, (1, decoder_h, decoder_w, 1))  # [B*T*num_slots, decoder_h, decoder_w, slot_dim]
    
    # Add positional encoding
    x = self.sub('dec_pos_embed', ImagePositionalEmbedding, (decoder_h, decoder_w),
                 feature_dim=slot_dim, **self.kw)(x)
    
    # Use standard decoder approach: upsample with resize then conv
    # Keep in channel-last format (NHWC) for easier manipulation
    # x: [B*T*num_slots, decoder_h, decoder_w, slot_dim]
    
    K = self.kernel
    for i, depth in enumerate(reversed(self.depths)):
      # Upsample spatial dimensions by 2x
      batch, h_cur, w_cur, c_cur = x.shape
      x = jax.image.resize(x, 
                          shape=(batch, h_cur * 2, w_cur * 2, c_cur),
                          method='bilinear')
      
      # Apply conv (channel-last, NHWC format)
      x = self.sub(f'deconv{i}', nn.Conv2D, depth, K, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'deconv{i}norm', nn.Norm, self.norm)(x))
    
    # Final layer: project to RGB + mask (4 channels)
    x = self.sub('dec_final', nn.Conv2D, 4, K, **self.kw)(x)
    # x: [B*T*N, H, W, 4]
    
    # Split into RGB and mask
    rgbs = x[..., :3]  # [B*T*N, H, W, 3]
    mask_logits = x[..., 3:4]  # [B*T*N, H, W, 1]
    
    # Reshape to separate slots
    rgbs = rgbs.reshape((batch_time, num_slots, h, w, 3))
    mask_logits = mask_logits.reshape((batch_time, num_slots, h, w, 1))
    
    # Softmax over slots to get masks
    masks = jax.nn.softmax(mask_logits, axis=1)
    
    # Composite: weighted sum over slots
    reconstruction = jnp.sum(rgbs * masks, axis=1)  # [B*T, H, W, 3]
    reconstruction = jnp.clip(reconstruction, 0, 1)
    
    # Reshape back to original batch/time structure
    reconstruction = reconstruction.reshape((*bshape, h, w, 3))
    
    # Convert to output format (similar to standard decoder)
    recons = {}
    offset = 0
    for key in sorted(self.imgkeys):
      shape = self.obs_space[key].shape
      channels = shape[2]
      recon_key = reconstruction[..., offset:offset+channels]
      
      # Convert to MSE output and aggregate over spatial dimensions
      out = embodied.jax.outs.MSE(recon_key)
      out = embodied.jax.outs.Agg(out, 3, jnp.sum)  # Aggregate over H, W, C dimensions
      recons[key] = out
      offset += channels
    
    entries = {}
    return carry, entries, recons

