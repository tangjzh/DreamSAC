import numpy as np

x = np.load('data/bair_preprocessed/test/traj_00000.npz')
print(x['image'].shape)
print(x.keys())