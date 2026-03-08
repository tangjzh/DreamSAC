"""
Configuration for domain randomization transfer learning experiments.

This script generates parameter sets and provides utilities for running
experiments with different physics configurations.
"""

import numpy as np
import pickle
import pathlib


# Define physics parameter ranges for randomization
PHYSICS_PARAMS = {
    'body_mass_range': (0.5, 1.5),      # Body masses: 50%-150% of original
    'friction_range': (0.7, 1.3),        # Friction: 70%-130% of original
    'damping_range': (0.8, 1.2),         # Damping: 80%-120% of original
    'actuator_gain_range': (0.8, 1.2),   # Actuator gains: 80%-120% of original
}


def generate_and_save_param_sets(
    env_name='walker_walk',
    num_sets=100,
    train_ratio=0.8,
    seed=42,
    output_dir='./param_sets',
):
  """
  Generate and save parameter sets for experiments.
  
  Args:
    env_name: Environment name (e.g., 'walker_walk', 'cheetah_run')
    num_sets: Total number of parameter sets
    train_ratio: Ratio of sets for training (e.g., 0.8 for 80%)
    seed: Random seed
    output_dir: Output directory for parameter sets
  
  Returns:
    Dictionary with train and test parameter set indices
  """
  from embodied.envs.dmc_randomized import generate_param_sets
  
  output_dir = pathlib.Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Generate parameter sets
  param_sets = generate_param_sets(
      base_env_name=env_name,
      num_sets=num_sets,
      physics_params=PHYSICS_PARAMS,
      seed=seed,
  )
  
  # Split into train and test sets
  np.random.seed(seed)
  indices = np.arange(num_sets)
  np.random.shuffle(indices)
  
  num_train = int(num_sets * train_ratio)
  train_indices = sorted(indices[:num_train].tolist())
  test_indices = sorted(indices[num_train:].tolist())
  
  # Save parameter sets
  data = {
      'param_sets': param_sets,
      'train_indices': train_indices,
      'test_indices': test_indices,
      'env_name': env_name,
      'physics_params': PHYSICS_PARAMS,
      'seed': seed,
  }
  
  output_file = output_dir / f'{env_name}_params.pkl'
  with open(output_file, 'wb') as f:
    pickle.dump(data, f)
  
  print(f"Generated {num_sets} parameter sets for {env_name}")
  print(f"  Train sets: {len(train_indices)} ({train_ratio*100:.0f}%)")
  print(f"  Test sets:  {len(test_indices)} ({(1-train_ratio)*100:.0f}%)")
  print(f"  Saved to: {output_file}")
  print(f"\nTrain indices: {train_indices}")
  print(f"Test indices:  {test_indices}")
  
  return data


def load_param_sets(env_name='walker_walk', param_dir='./param_sets'):
  """Load parameter sets from file."""
  param_dir = pathlib.Path(param_dir)
  param_file = param_dir / f'{env_name}_params.pkl'
  
  with open(param_file, 'rb') as f:
    data = pickle.load(f)
  
  return data


if __name__ == '__main__':
  # Example: Generate parameter sets for different environments
  
  # Walker Walk
  # generate_and_save_param_sets(
  #     env_name='walker_walk',
  #     num_sets=100,
  #     train_ratio=0.8,
  #     seed=42,
  # )
  
  # # Cheetah Run
  # generate_and_save_param_sets(
  #     env_name='cheetah_run',
  #     num_sets=100,
  #     train_ratio=0.8,
  #     seed=42,
  # )
  
  # # Hopper Hop
  # generate_and_save_param_sets(
  #     env_name='hopper_hop',
  #     num_sets=100,
  #     train_ratio=0.8,
  #     seed=42,
  # )

#   generate_and_save_param_sets(
#       env_name='reacher_hard',
#       num_sets=100,
#       train_ratio=0.8,
#       seed=42,
#  )

  generate_and_save_param_sets(
        env_name='humanoid_walk',
        num_sets=100,
        train_ratio=0.8,
        seed=42,
  )