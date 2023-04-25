import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from os.path import isfile, join
from os import listdir
from pathlib import Path
import struct
from PIL import Image

class ImageDataset(Dataset):
    """ 
        Dataset class for stereo images. 
    """
    def __init__(self, dataset_dir: str = "data/NewTsukubaStereoDataset", image_size: tuple[int, int] = (480, 640)) -> None:
        
        self.dataset_dir = dataset_dir
        self.images_dir = join(self.dataset_dir, "images")
        self.depth_maps_dir = join(self.dataset_dir, "groundtruth/depth_maps")
        self.images_filenames = sorted(listdir(self.images_dir), key=lambda x: int(x[2:-4]))
        self.length = len(self.images_filenames) // 2
        
        self.transforms = Compose([ToTensor(), Resize(size=image_size, antialias=False)])
        self.depth_map_transforms = Resize(size=image_size, antialias=False)

        # Mean and standard deviations for left and right views computed on training data
        # TODO: Update these values
        self.means = [(0.3998, 0.5025, 0.5001), (0.3980, 0.5005, 0.4981)]
        self.stds = [(0.2175, 0.2275, 0.2347), (0.2214, 0.2236, 0.2304)]
        self.normalize_left = Normalize(self.means[0], self.stds[0])
        self.normalize_right = Normalize(self.means[1], self.stds[1])

    def __len__(self) -> int:
        return self.length 

    def load_pfm(self, filename: str) -> torch.Tensor:
        """
            Load depth map from .pfm file, return as tensor of shape (1,2,H,W).
        """
        with Path(filename).open('rb') as pfm_file:
            line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
            assert line1 in ('PF', 'Pf')
            channels = 3 if "PF" in line1 else 1
            width, height = (int(s) for s in line2.split())
            scale_endianess = float(line3)
            bigendian = scale_endianess > 0
            scale = abs(scale_endianess)
            buffer = pfm_file.read()
            samples = width * height * channels
            assert len(buffer) == samples * 4
            fmt = f'{"<>"[bigendian]}{samples}f'
            decoded = struct.unpack(fmt, buffer)
            shape = (height, width, 3) if channels == 3 else (height, width)
            out = np.flipud(np.reshape(decoded, shape)) * scale
            return torch.Tensor(out).unsqueeze(0)

    def load_depth_map(self, idx: int) -> torch.Tensor:
        """
            Load left and right depth maps from pfm files.
            Returns tensor of shape (1, 2, H, W)
        """
        left_view = self.load_pfm(join(self.depth_maps_dir, f"L_{idx+1:05}.pfm"))
        right_view = self.load_pfm(join(self.depth_maps_dir, f"R_{idx+1:05}.pfm"))
        return torch.stack([self.depth_map_transforms(left_view), self.depth_map_transforms(right_view)], dim=1)

    def load_stereo_image(self, idx: int) -> torch.Tensor:
        """
            Load stereo image from png files.
            Return tensor of shape (3, 2, H, W)
        """
        left_view = Image.open(join(self.images_dir, f"L_{idx+1:05}.png")).convert("RGB")
        right_view = Image.open(join(self.images_dir, f"R_{idx+1:05}.png")).convert("RGB")
        return torch.stack([self.transforms(left_view), self.transforms(right_view)], dim=1)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Returns a stereo image of shape (3, 2, H, W) and ground truth depth map of shape (2, H, W).
        """
        image = self.load_stereo_image(idx)
        gt = self.load_depth_map(idx)
        image[:,0] = self.normalize_left(image[:,0])
        image[:,1] = self.normalize_right(image[:,1])
        return (image, gt)

    def label_str(self, label: int) -> str:
        return self.label_list[label]


def seed_worker(worker_id):
    """
        From PyTorch docs. Ensures deterministic dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloaders(batch_size: int, test: float, val: float, image_size: tuple[int, int] = (480, 640), 
                        random_seed: int = 42, num_workers: int = 8) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
        Create data loaders for training, validation and test datasets.
    """
    dataset = ImageDataset(image_size=image_size)
    generator = torch.Generator().manual_seed(random_seed)

    val_size = int(val * len(dataset))
    test_size = int(test * len(dataset))
    train_size = len(dataset) - val_size - test_size

    #train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_ds = Subset(dataset, torch.arange(0, train_size))
    val_ds = Subset(dataset, torch.arange(train_size, train_size + val_size))
    test_ds = Subset(dataset, torch.arange(train_size + val_size, len(dataset)))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, num_workers=num_workers)

    return (train_dl, val_dl, test_dl)
