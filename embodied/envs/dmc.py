import functools
import os

import elements
import embodied
import numpy as np

# dm_control selects its renderer at import time.
os.environ.setdefault('MUJOCO_GL', 'egl')
from dm_control import manipulation
from dm_control import suite
from dm_control.locomotion.examples import basic_rodent_2020

from . import from_dm


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), proprio=True, image=True, camera=-1,
      seed=0, gravity_scale=1.0, friction_scale=1.0):
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        env = manipulation.load(task + '_vision')
      elif domain == 'rodent':
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        env = getattr(basic_rodent_2020, task)()
      else:
        env = suite.load(domain, task, task_kwargs={'random': seed})
    if gravity_scale <= 0 or friction_scale <= 0:
      raise ValueError('Physics scales must be positive.')
    env.physics.model.opt.gravity[:] *= gravity_scale
    env.physics.model.geom_friction[:] *= friction_scale
    self._dmenv = env
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size
    self._proprio = proprio
    self._image = image
    self._camera = camera

  def close(self):
    self._dmenv.close()

  @functools.cached_property
  def obs_space(self):
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    spaces = self._env.obs_space.copy()
    if not self._proprio:
      spaces = {k: spaces[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    spaces[key] = elements.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    if not self._proprio:
      obs = {k: obs[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs
