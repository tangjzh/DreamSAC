import functools
import os
import copy

import elements
import embodied
import numpy as np
from dm_control import suite
from dm_control.suite.utils import randomizers

from . import dmc


class DMCRandomized(dmc.DMC):
  """
  DMC environment with randomized physics parameters.
  Supports training on multiple parameter configurations for domain randomization
  and transfer learning experiments.
  """

  def __init__(
      self,
      env,
      repeat=1,
      size=(64, 64),
      proprio=True,
      image=True,
      camera=-1,
      physics_params=None,
      param_set_id=None,
      all_param_sets=None,
      seed=0,
      gravity_scale=1.0,
      friction_scale=1.0,
  ):
    """
    Args:
      env: Environment name (e.g., 'walker_walk')
      repeat: Action repeat
      size: Image size
      proprio: Include proprioceptive observations
      image: Include image observations
      camera: Camera ID
      physics_params: Dict of physics parameters to randomize
                     e.g., {'body_mass_range': (0.5, 1.5), 
                            'friction_range': (0.5, 1.5),
                            'damping_range': (0.8, 1.2)}
      param_set_id: Specific parameter set ID (for evaluation)
      all_param_sets: Pre-defined list of parameter sets for reproducibility
    """
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'

    self._physics_params = physics_params or {}
    self._param_set_id = param_set_id
    self._all_param_sets = all_param_sets
    self._current_params = None

    # Load base environment
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if domain == 'cup':
        domain = 'ball_in_cup'
      
      # Create environment with custom physics
      env_obj = suite.load(domain, task, task_kwargs={'random': seed})
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      
      # Apply physics randomization
      self._apply_physics_randomization(env_obj)
      
      super().__init__(
          env_obj, repeat=repeat, size=size,
          proprio=proprio, image=image, camera=camera,
          gravity_scale=gravity_scale, friction_scale=friction_scale)
    else:
      # If env is already an object
      self._apply_physics_randomization(env)
      super().__init__(
          env, repeat=repeat, size=size,
          proprio=proprio, image=image, camera=camera,
          gravity_scale=gravity_scale, friction_scale=friction_scale)

  def _apply_physics_randomization(self, env):
    """Apply physics parameter randomization to the environment."""
    physics = env.physics
    
    # If using predefined parameter sets
    if self._all_param_sets is not None and self._param_set_id is not None:
      params = self._all_param_sets[self._param_set_id]
      self._current_params = params
      self._set_physics_params(physics, params)
      return
    
    # Otherwise, sample random parameters
    if not self._physics_params:
      return
    
    params = {}
    
    # Randomize body masses
    if 'body_mass_range' in self._physics_params:
      mass_range = self._physics_params['body_mass_range']
      original_masses = physics.model.body_mass.copy()
      mass_multipliers = np.random.uniform(
          mass_range[0], mass_range[1], size=len(original_masses))
      params['body_mass'] = original_masses * mass_multipliers
    
    # Randomize friction coefficients
    if 'friction_range' in self._physics_params:
      friction_range = self._physics_params['friction_range']
      original_friction = physics.model.geom_friction.copy()
      friction_multipliers = np.random.uniform(
          friction_range[0], friction_range[1], 
          size=original_friction.shape)
      params['geom_friction'] = original_friction * friction_multipliers
    
    # Randomize damping
    if 'damping_range' in self._physics_params:
      damping_range = self._physics_params['damping_range']
      original_damping = physics.model.dof_damping.copy()
      damping_multipliers = np.random.uniform(
          damping_range[0], damping_range[1], 
          size=len(original_damping))
      params['dof_damping'] = original_damping * damping_multipliers
    
    # Randomize joint stiffness
    if 'stiffness_range' in self._physics_params:
      stiffness_range = self._physics_params['stiffness_range']
      original_stiffness = physics.model.jnt_stiffness.copy()
      stiffness_multipliers = np.random.uniform(
          stiffness_range[0], stiffness_range[1],
          size=len(original_stiffness))
      params['jnt_stiffness'] = original_stiffness * stiffness_multipliers
    
    # Randomize actuator gains
    if 'actuator_gain_range' in self._physics_params:
      gain_range = self._physics_params['actuator_gain_range']
      original_gains = physics.model.actuator_gainprm.copy()
      gain_multipliers = np.random.uniform(
          gain_range[0], gain_range[1],
          size=original_gains.shape[0])
      params['actuator_gainprm'] = original_gains.copy()
      params['actuator_gainprm'][:, 0] *= gain_multipliers
    
    self._current_params = params
    self._set_physics_params(physics, params)
  
  def _set_physics_params(self, physics, params):
    """Set physics parameters from a parameter dictionary."""
    for key, value in params.items():
      # Special handling for gravity (stored in physics.model.opt)
      if key == 'gravity':
        physics.model.opt.gravity[:] = value
      elif hasattr(physics.model, key):
        setattr(physics.model, key, value)

  def get_current_params(self):
    """Return the current physics parameters."""
    return copy.deepcopy(self._current_params)


def generate_param_sets(
    base_env_name,
    num_sets=10,
    physics_params=None,
    seed=0,
):
  """
  Generate multiple parameter sets for reproducible experiments.
  
  Args:
    base_env_name: Base environment name (e.g., 'walker_walk')
    num_sets: Number of parameter sets to generate
    physics_params: Dictionary of parameter ranges
    seed: Random seed for reproducibility
  
  Returns:
    List of parameter dictionaries
  """
  np.random.seed(seed)
  
  # Load a temporary environment to get default parameters
  domain, task = base_env_name.split('_', 1)
  if domain == 'cup':
    domain = 'ball_in_cup'
  temp_env = suite.load(domain, task)
  physics = temp_env.physics
  
  param_sets = []
  
  for i in range(num_sets):
    params = {}
    
    if 'body_mass_range' in physics_params:
      mass_range = physics_params['body_mass_range']
      original_masses = physics.model.body_mass.copy()
      mass_multipliers = np.random.uniform(
          mass_range[0], mass_range[1], size=len(original_masses))
      params['body_mass'] = original_masses * mass_multipliers
    
    if 'friction_range' in physics_params:
      friction_range = physics_params['friction_range']
      original_friction = physics.model.geom_friction.copy()
      friction_multipliers = np.random.uniform(
          friction_range[0], friction_range[1],
          size=original_friction.shape)
      params['geom_friction'] = original_friction * friction_multipliers
    
    if 'damping_range' in physics_params:
      damping_range = physics_params['damping_range']
      original_damping = physics.model.dof_damping.copy()
      damping_multipliers = np.random.uniform(
          damping_range[0], damping_range[1],
          size=len(original_damping))
      params['dof_damping'] = original_damping * damping_multipliers
    
    if 'stiffness_range' in physics_params:
      stiffness_range = physics_params['stiffness_range']
      original_stiffness = physics.model.jnt_stiffness.copy()
      stiffness_multipliers = np.random.uniform(
          stiffness_range[0], stiffness_range[1],
          size=len(original_stiffness))
      params['jnt_stiffness'] = original_stiffness * stiffness_multipliers
    
    if 'actuator_gain_range' in physics_params:
      gain_range = physics_params['actuator_gain_range']
      original_gains = physics.model.actuator_gainprm.copy()
      gain_multipliers = np.random.uniform(
          gain_range[0], gain_range[1],
          size=original_gains.shape[0])
      params['actuator_gainprm'] = original_gains.copy()
      params['actuator_gainprm'][:, 0] *= gain_multipliers
    
    param_sets.append(params)
  
  return param_sets

