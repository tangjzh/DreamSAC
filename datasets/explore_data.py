import numpy as np
import cv2
import imageio

# Load the data
x = np.load('/mnt/HDD_16TB/jinzhou/Ego4D/v2/ego4d_processed/test/0ed57e80-0e57-47d3-8942-54450722dc95_clip_62.npz', allow_pickle=True)

# Extract the images
images = x['images']
print(images.shape)
print(x['instruction'])

# Convert images to uint8 if necessary
if images.dtype != np.uint8:
    images = (255 * (images - images.min()) / (images.max() - images.min())).astype(np.uint8)

# Prepare to save the gif
gif_path = 'output_video.gif'

# print(images)

# Save as a GIF
imageio.mimsave(gif_path, images, fps=8)

print(f"GIF saved at {gif_path}")