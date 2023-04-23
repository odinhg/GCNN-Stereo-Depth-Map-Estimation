from utils import ImageDataset, create_dataloaders
from utils import visualize_stereo_depth_map, visualize_stereo_image 
import torch
from PIL import Image

dl, _, _ = create_dataloaders(batch_size=4, val=0.2, test=0.2)

"""
# Test data loader and visualization of stereo images and depth maps
for data in dl:
    image, gt = data[0][2], data[1][2]
    print(image.shape)
    print(gt.shape)
    visualize_stereo_image(image)
    visualize_stereo_depth_map(gt)
    exit()
"""
