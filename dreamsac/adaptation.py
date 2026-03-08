import jax
import jax.numpy as jnp
import ninjax as nj
from typing import Dict, List, Optional


class AdaptationStrategy:
  """
  Base class for adaptation strategies.
  
  Determines which components to freeze/train and with what learning rates.
  """
  
  def __init__(self, mode: str, config: Dict):
    """
    Args:
      mode: 'zero_shot' or 'parametric'
      config: Configuration dict with learning rates, etc.
    """
    self.mode = mode
    self.config = config
  
  def get_trainable_components(self) -> Dict[str, bool]:
    """
    Returns dict indicating which components should be trained.
    
    Returns:
      {
        'encoder': bool,
        'decoder': bool,
        'dynamics': bool,
        'policy': bool,
        'critic': bool,
      }
    """
    raise NotImplementedError
  
  def get_learning_rate_scales(self) -> Dict[str, float]:
    """
    Returns learning rate multipliers for each component.
    
    Returns:
      {
        'encoder': float,  # 0.0 = frozen, 1.0 = full LR, 0.1 = 10% LR
        'decoder': float,
        'dynamics': float,
        'policy': float,
        'critic': float,
      }
    """
    raise NotImplementedError


class ZeroShotAdaptation(AdaptationStrategy):
  """
  Zero-shot structural generalization strategy.
  
  For tasks with novel initial poses, orientations, or camera viewpoints,
  but same physics parameters (mass, gravity, etc.).
  
  Strategy:
  - Freeze: encoder, decoder, dynamics (entire world model)
  - Train: policy, critic
  - Rationale: World model already captures correct physics, just need new policy
  """
  
  def __init__(self, config: Dict):
    super().__init__('zero_shot', config)
  
  def get_trainable_components(self) -> Dict[str, bool]:
    return {
        'encoder': False,      # Frozen
        'decoder': False,      # Frozen
        'dynamics': False,     # Frozen
        'policy': True,        # Train from scratch
        'critic': True,        # Train from scratch
    }
  
  def get_learning_rate_scales(self) -> Dict[str, float]:
    return {
        'encoder': 0.0,        # No updates
        'decoder': 0.0,        # No updates
        'dynamics': 0.0,       # No updates
        'policy': 1.0,         # Full learning rate
        'critic': 1.0,         # Full learning rate
    }


class ParametricAdaptation(AdaptationStrategy):
  """
  Rapid parametric adaptation strategy.
  
  For tasks with different physics parameters (new gravity, mass, friction, etc.)
  but same visual and structural properties.
  
  Strategy:
  - Freeze: encoder (visual features unchanged)
  - Fine-tune: dynamics (Hamiltonian H, input matrix g) with small LR
  - Train: policy, critic (from scratch or continue)
  - Rationale: Invariant structure helps quickly adapt implicit parameters
  """
  
  def __init__(self, config: Dict):
    super().__init__('parametric', config)
    
    # Small learning rate for dynamics fine-tuning
    self.dynamics_lr_scale = config.get('dynamics_lr_scale', 0.1)
    
    # Whether to reset policy/critic or continue training
    self.reset_policy = config.get('reset_policy', True)
  
  def get_trainable_components(self) -> Dict[str, bool]:
    return {
        'encoder': False,      # Frozen (visual unchanged)
        'decoder': True,       # Can fine-tune (optional)
        'dynamics': True,      # Fine-tune with small LR
        'policy': True,        # Train
        'critic': True,        # Train
    }
  
  def get_learning_rate_scales(self) -> Dict[str, float]:
    return {
        'encoder': 0.0,                      # No updates
        'decoder': self.dynamics_lr_scale,   # Small LR (optional)
        'dynamics': self.dynamics_lr_scale,  # Small LR (e.g., 0.1x)
        'policy': 1.0,                       # Full LR
        'critic': 1.0,                       # Full LR
    }


class AdaptationManager:
  """
  Manages adaptation strategies and applies them to the agent.
  """
  
  def __init__(self, strategy: AdaptationStrategy):
    self.strategy = strategy
  
  @staticmethod
  def create(mode: str, config: Dict) -> 'AdaptationManager':
    """
    Factory method to create adaptation manager.
    
    Args:
      mode: 'zero_shot' or 'parametric'
      config: Configuration dict
    
    Returns:
      AdaptationManager instance
    """
    if mode == 'zero_shot':
      strategy = ZeroShotAdaptation(config)
    elif mode == 'parametric':
      strategy = ParametricAdaptation(config)
    else:
      raise ValueError(f"Unknown adaptation mode: {mode}")
    
    return AdaptationManager(strategy)
  
  def apply_to_optimizer(self, optimizers: Dict, base_lr: float = 1.0):
    """
    Apply learning rate scales to optimizers.
    
    Args:
      optimizers: Dict of {component_name: optimizer}
      base_lr: Base learning rate
    
    Returns:
      Modified optimizers with scaled learning rates
    """
    lr_scales = self.strategy.get_learning_rate_scales()
    
    scaled_optimizers = {}
    for name, opt in optimizers.items():
      if name in lr_scales:
        scale = lr_scales[name]
        if scale == 0.0:
          # Frozen: use zero learning rate
          scaled_optimizers[name] = freeze_optimizer(opt)
        elif scale != 1.0:
          # Scaled: multiply learning rate
          scaled_optimizers[name] = scale_optimizer_lr(opt, scale)
        else:
          # Full LR: keep as is
          scaled_optimizers[name] = opt
      else:
        scaled_optimizers[name] = opt
    
    return scaled_optimizers
  
  def should_train_component(self, component: str) -> bool:
    """Check if a component should be trained."""
    trainable = self.strategy.get_trainable_components()
    return trainable.get(component, True)
  
  def get_parameter_mask(self, params: Dict, component_map: Dict) -> Dict:
    """
    Create a mask for parameters indicating which should be updated.
    
    Args:
      params: Parameter pytree
      component_map: Mapping from param path to component name
    
    Returns:
      Mask pytree (same structure as params, values are 0.0 or 1.0)
    """
    trainable = self.strategy.get_trainable_components()
    
    def create_mask(path, param):
      # Determine which component this parameter belongs to
      component = component_map.get(path, 'other')
      
      # Return mask value
      if component in trainable:
        return 1.0 if trainable[component] else 0.0
      else:
        return 1.0  # Default: trainable
    
    mask = jax.tree_util.tree_map_with_path(
        lambda path, param: jnp.ones_like(param) * create_mask(path, param),
        params
    )
    
    return mask


def freeze_optimizer(optimizer):
  """
  Create a frozen optimizer that doesn't update parameters.
  
  Args:
    optimizer: Original optimizer
  
  Returns:
    Optimizer with zero learning rate
  """
  import optax
  
  # Chain with zero scaling
  return optax.chain(
      optax.scale(0.0),
      optimizer
  )


def scale_optimizer_lr(optimizer, scale: float):
  """
  Scale the learning rate of an optimizer.
  
  Args:
    optimizer: Original optimizer
    scale: Scaling factor (e.g., 0.1 for 10% of original LR)
  
  Returns:
    Optimizer with scaled learning rate
  """
  import optax
  
  # Chain with scaling
  return optax.chain(
      optax.scale(scale),
      optimizer
  )


def create_adaptation_strategy(config: Dict) -> Optional[AdaptationManager]:
  """
  Create adaptation strategy from config.
  
  Args:
    config: Configuration dict with 'adaptation' section
  
  Returns:
    AdaptationManager or None if not using adaptation
  """
  if 'adaptation' not in config:
    return None
  
  adaptation_config = config['adaptation']
  mode = adaptation_config.get('mode', None)
  
  if mode is None:
    return None
  
  return AdaptationManager.create(mode, adaptation_config)


# ============================================================================
# Helper functions for component identification
# ============================================================================

def identify_component(param_path: str) -> str:
  """
  Identify which component a parameter belongs to based on its path.
  
  Args:
    param_path: Parameter path string (e.g., 'encoder/conv0/kernel')
  
  Returns:
    Component name: 'encoder', 'decoder', 'dynamics', 'policy', 'critic', or 'other'
  """
  path_lower = param_path.lower()
  
  if 'encoder' in path_lower or 'enc/' in path_lower:
    return 'encoder'
  elif 'decoder' in path_lower or 'dec/' in path_lower:
    return 'decoder'
  elif 'dynamics' in path_lower or 'rssm' in path_lower or 'hamiltonian' in path_lower:
    return 'dynamics'
  elif 'policy' in path_lower or 'actor' in path_lower:
    return 'policy'
  elif 'critic' in path_lower or 'value' in path_lower:
    return 'critic'
  else:
    return 'other'


def create_component_map(params: Dict) -> Dict[str, str]:
  """
  Create mapping from parameter paths to component names.
  
  Args:
    params: Parameter pytree
  
  Returns:
    Dict mapping param paths to component names
  """
  component_map = {}
  
  def map_path(path, param):
    path_str = '/'.join([str(p) for p in path])
    component_map[path_str] = identify_component(path_str)
  
  jax.tree_util.tree_map_with_path(map_path, params)
  
  return component_map

