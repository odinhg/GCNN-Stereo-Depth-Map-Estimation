import torch.nn as nn
from models import GUNetModel 
from utils import create_dataloaders, Group, d2_r, d2_mh, d2_mv, d2_e, visualize_stereo_depth_map, visualize_stereo_image 
from config import val_fraction, test_fraction

functions = [d2_e, d2_r, d2_mh, d2_mv]
cayley_table = [[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]
group = Group(functions, cayley_table)

dl, _, _ = create_dataloaders(batch_size=2, val=val_fraction, test=test_fraction)

model = GUNetModel(group)
model.eval()

for batch in dl:
    images, _ = batch
    break

k = 1 # Index for image to use from batch
x = images[k]
out = model(images)[k]

visualize_stereo_image(x)
visualize_stereo_depth_map(out)

"""
for g in functions:
    gx = g(x)
    gout = model(g(images))[k]
    gx, gout = normalize_tensor(gx), normalize_tensor(gout)
    visualize_tensor(gx, filename=f"docs/input_image_{g.__name__}.png")
    visualize_tensor(gout, filename=f"docs/output_{g.__name__}.png")
"""
