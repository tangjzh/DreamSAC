import jax
import jax.numpy as jnp
import ninjax as nj
from typing import Dict, Tuple

f32 = jnp.float32


def random_perspective_transform(images, max_shift=0.1, key=None):
  """
  Apply random perspective transformation to images.
  
  Args:
    images: [B, H, W, C] images in range [0, 1]
    max_shift: maximum shift ratio for perspective transform
    key: JAX random key
  
  Returns:
    transformed_images: [B, H, W, C]
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  
  batch_size, height, width, channels = images.shape
  
  # Generate random perspective parameters
  # We'll simulate this with affine transform + slight distortion
  key1, key2 = jax.random.split(key)
  
  # Random shifts for corners (simulating perspective)
  shifts = jax.random.uniform(key1, (batch_size, 4, 2), 
                             minval=-max_shift, maxval=max_shift)
  
  # Apply shift using jax.image operations
  # For simplicity, use random crop + resize to simulate perspective
  crop_h = int(height * (1 - max_shift))
  crop_w = int(width * (1 - max_shift))
  
  # Random crop - ensure indices are integers at trace time
  if crop_h >= height:
    crop_h = height
  if crop_w >= width:
    crop_w = width
    
  y_max = max(1, height - crop_h)
  x_max = max(1, width - crop_w)
  y_offset = jax.random.randint(key1, (batch_size,), 0, y_max)
  x_offset = jax.random.randint(key2, (batch_size,), 0, x_max)
  
  def crop_and_resize(img, y, x):
    # Ensure y, x are within bounds
    y_safe = jnp.clip(y, 0, y_max - 1)
    x_safe = jnp.clip(x, 0, x_max - 1)
    cropped = jax.lax.dynamic_slice(img, (y_safe, x_safe, 0), (crop_h, crop_w, channels))
    resized = jax.image.resize(cropped, (height, width, channels), method='bilinear')
    return resized
  
  transformed = jax.vmap(crop_and_resize)(images, y_offset, x_offset)
  
  return transformed


def random_camera_jitter(images, max_jitter=0.05, key=None):
  """
  Apply random camera jitter (small translations and rotations).
  
  Args:
    images: [B, H, W, C]
    max_jitter: maximum jitter ratio
    key: JAX random key
  
  Returns:
    jittered_images: [B, H, W, C]
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  
  batch_size, height, width, channels = images.shape
  
  # Random translation
  key1, key2 = jax.random.split(key)
  max_shift_pixels = int(height * max_jitter)
  
  dy = jax.random.randint(key1, (batch_size,), -max_shift_pixels, max_shift_pixels + 1)
  dx = jax.random.randint(key2, (batch_size,), -max_shift_pixels, max_shift_pixels + 1)
  
  def translate(img, dy_val, dx_val):
    # Use roll to shift image
    # Don't use int() - keep as JAX array for tracing
    shifted = jnp.roll(img, shift=dy_val, axis=0)
    shifted = jnp.roll(shifted, shift=dx_val, axis=1)
    return shifted
  
  jittered = jax.vmap(translate)(images, dy, dx)
  
  return jittered


def random_brightness_contrast(images, brightness_range=0.2, contrast_range=0.2, key=None):
  """
  Random brightness and contrast adjustment.
  
  Args:
    images: [B, H, W, C] in range [0, 1]
    brightness_range: range for brightness adjustment
    contrast_range: range for contrast adjustment
    key: JAX random key
  
  Returns:
    adjusted_images: [B, H, W, C]
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  
  batch_size = images.shape[0]
  
  # Random brightness and contrast factors
  key1, key2 = jax.random.split(key)
  brightness = jax.random.uniform(key1, (batch_size, 1, 1, 1), 
                                 minval=1 - brightness_range, 
                                 maxval=1 + brightness_range)
  contrast = jax.random.uniform(key2, (batch_size, 1, 1, 1),
                               minval=1 - contrast_range,
                               maxval=1 + contrast_range)
  
  # Apply: I' = contrast * I + brightness
  adjusted = images * contrast + (brightness - 1)
  adjusted = jnp.clip(adjusted, 0, 1)
  
  return adjusted


def apply_viewpoint_augmentation(images, key=None, 
                                 perspective=True, 
                                 jitter=True, 
                                 color=True):
  """
  Apply combined viewpoint augmentations.
  
  Args:
    images: [B, H, W, C] in range [0, 1]
    key: JAX random key
    perspective: apply perspective transform
    jitter: apply camera jitter
    color: apply color jittering
  
  Returns:
    augmented_images: [B, H, W, C]
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  
  augmented = images
  
  if perspective:
    key, subkey = jax.random.split(key)
    augmented = random_perspective_transform(augmented, max_shift=0.1, key=subkey)
  
  if jitter:
    key, subkey = jax.random.split(key)
    augmented = random_camera_jitter(augmented, max_jitter=0.05, key=subkey)
  
  if color:
    key, subkey = jax.random.split(key)
    augmented = random_brightness_contrast(augmented, 
                                          brightness_range=0.2,
                                          contrast_range=0.2, 
                                          key=subkey)
  
  return augmented


def cosine_similarity(x, y, eps=1e-8):
  """
  Compute cosine similarity between two tensors.
  
  Args:
    x: [B, D]
    y: [B, D]
  
  Returns:
    similarity: [B]
  """
  x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True) + eps
  y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True) + eps
  
  x_normalized = x / x_norm
  y_normalized = y / y_norm
  
  similarity = jnp.sum(x_normalized * y_normalized, axis=-1)
  return similarity


def infonce_loss(z_anchor, z_positive, temperature=0.1):
  """
  InfoNCE contrastive loss.
  
  Args:
    z_anchor: [B, D] anchor representations
    z_positive: [B, D] positive (augmented view) representations
    temperature: temperature parameter for softmax
  
  Returns:
    loss: scalar
  """
  batch_size = z_anchor.shape[0]
  
  # Normalize representations
  z_anchor = z_anchor / (jnp.linalg.norm(z_anchor, axis=-1, keepdims=True) + 1e-8)
  z_positive = z_positive / (jnp.linalg.norm(z_positive, axis=-1, keepdims=True) + 1e-8)
  
  # Compute similarity matrix: [B, B]
  # similarity[i, j] = z_anchor[i] · z_positive[j]
  similarity_matrix = jnp.matmul(z_anchor, z_positive.T) / temperature
  
  # Positive pairs are on the diagonal
  # Negatives are all off-diagonal elements
  labels = jnp.arange(batch_size)
  
  # Cross-entropy loss with labels as diagonal indices
  loss = -jnp.mean(jax.nn.log_softmax(similarity_matrix, axis=1)[labels, labels])
  
  return loss


def compute_viewpoint_robustness_loss(encoder_fn, images, key=None, temperature=0.1):
  """
  Compute viewpoint robustness loss L_vr.
  
  Creates two augmented views of each image and applies InfoNCE loss
  to encourage viewpoint-invariant representations.
  
  Args:
    encoder_fn: function that maps images -> latent representations
                Takes [B, H, W, C] -> [B, D]
    images: [B, H, W, C] original images
    key: JAX random key
    temperature: temperature for InfoNCE
  
  Returns:
    loss: scalar viewpoint robustness loss
    metrics: dict with auxiliary metrics
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  
  # Generate two augmented views
  key1, key2 = jax.random.split(key)
  images_A = apply_viewpoint_augmentation(images, key=key1)
  images_B = apply_viewpoint_augmentation(images, key=key2)
  
  # Encode both views
  z_A = encoder_fn(images_A)
  z_B = encoder_fn(images_B)
  
  # Compute InfoNCE loss
  loss = infonce_loss(z_A, z_B, temperature=temperature)
  
  # Compute metrics (using stop_gradient properly)
  z_A_stopped = jax.lax.stop_gradient(z_A)
  z_B_stopped = jax.lax.stop_gradient(z_B)
  
  # Positive pair similarity
  pos_sim = cosine_similarity(z_A_stopped, z_B_stopped).mean()
  
  # Negative pair similarity (off-diagonal)
  batch_size = z_A.shape[0]
  all_sim = jnp.matmul(z_A_stopped, z_B_stopped.T)
  mask = 1 - jnp.eye(batch_size)
  neg_sim = (all_sim * mask).sum() / (batch_size * (batch_size - 1) + 1e-8)
  
  metrics = {
    'vr_loss': loss,
    'vr_pos_sim': pos_sim,
    'vr_neg_sim': neg_sim,
  }
  
  return loss, metrics


def compute_slot_viewpoint_robustness_loss(slot_encoder_fn, images, actions, 
                                          prev_slots, reset, key=None, 
                                          temperature=0.1):
  """
  Viewpoint robustness loss adapted for slot-based encoder.
  
  Args:
    slot_encoder_fn: function (images, actions, prev_slots, reset) -> slots
                     Returns slots of shape [B, num_slots, slot_dim]
    images: [B, H, W, C]
    actions: [B, action_dim]
    prev_slots: [B, num_slots, slot_dim] or None
    reset: [B] boolean
    key: JAX random key
    temperature: temperature for InfoNCE
  
  Returns:
    loss: scalar
    metrics: dict
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  
  # Generate two augmented views
  key1, key2 = jax.random.split(key)
  images_A = apply_viewpoint_augmentation(images, key=key1)
  images_B = apply_viewpoint_augmentation(images, key=key2)
  
  # Encode both views to slots
  slots_A = slot_encoder_fn(images_A, actions, prev_slots, reset)
  slots_B = slot_encoder_fn(images_B, actions, prev_slots, reset)
  
  # Flatten slots to vectors: [B, num_slots * slot_dim]
  batch_size = slots_A.shape[0]
  z_A = slots_A.reshape((batch_size, -1))
  z_B = slots_B.reshape((batch_size, -1))
  
  # Compute InfoNCE loss
  loss = infonce_loss(z_A, z_B, temperature=temperature)
  
  # Metrics (using stop_gradient properly)
  z_A_stopped = jax.lax.stop_gradient(z_A)
  z_B_stopped = jax.lax.stop_gradient(z_B)
  
  pos_sim = cosine_similarity(z_A_stopped, z_B_stopped).mean()
  all_sim = jnp.matmul(z_A_stopped, z_B_stopped.T)
  mask = 1 - jnp.eye(batch_size)
  neg_sim = (all_sim * mask).sum() / (batch_size * (batch_size - 1) + 1e-8)
  
  metrics = {
    'vr_loss': loss,
    'vr_pos_sim': pos_sim,
    'vr_neg_sim': neg_sim,
  }
  
  return loss, metrics

