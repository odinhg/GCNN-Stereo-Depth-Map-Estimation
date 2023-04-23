import numpy as np
import torch
from PIL import Image
import random

def seed_random_generators(seed: int = 0, deterministic: bool = True) -> None:
    """
        Seed random generators with given seed. 
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())

def choose_model(configs):
    """ 
    Ask user to choose model/config to use.
    """
    n = -1
    while n < 0 or n >= len(configs):
        print("Select model:")
        for i, config in enumerate(configs):
            print(f"[{i}] {config['name']}")
        n = int(input() or 0)

    config = configs[n]
    print("Loaded model configuration:")
    for key, value in config.items():
        print(f"\t* {key}: {value}")

    return config

def visualize_stereo_image(x: torch.Tensor) -> None:
    """
        Visualize stereo image represented as tensor of shape (3, 2, H, W).
    """
    y = torch.cat([x[:,0], x[:,1]], dim=-1)
    y = torch.permute(y, (1, 2, 0))
    y = y.cpu().detach() * 255
    y = y.type(torch.uint8).numpy()
    image = Image.fromarray(y, mode="RGB")
    image.show()

def visualize_stereo_depth_map(x: torch.Tensor) -> None:
    """
        Visualize stereo depth map represented as tensor of shape (1, 2, H, W).
        Note that we normalize distances (originally in cm) to lie in the interval [0,1].
    """
    y = x.squeeze(0)
    y = torch.cat([y[0], y[1]], dim=-1)
    y = normalize_tensor(y) * 255
    y = y.cpu().detach()
    y = y.type(torch.uint8).numpy()
    image = Image.fromarray(y, mode="L")
    image.show()

