import cv2
import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from natsort import natsorted
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_frames')
TRAIN_SPLIT    = 0.75
VAL_SPLIT      = 0.15


# Ensure the root project directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RandomGlobalOrLocal(torch.nn.Module):
    """
    Custom transform that randomly switches between two strategies per sample:

    - LOCAL  (1 - global_prob): RandomCrop(crop_size)
        Teaches fine-grained texture and motion detail.

    - GLOBAL (global_prob):     Resize((crop_size, crop_size))
        Teaches the full spatial layout of the scene by squeezing the whole
        frame down to the target size. Aspect ratio is slightly distorted
        but global structure (roads, buildings, open areas) is preserved.

    Both paths produce the same output shape, so they can be mixed freely
    in the same DataLoader batch.
    """
    def __init__(self, crop_size: int, global_prob: float = 0.20):
        super().__init__()
        self.crop        = transforms.RandomCrop(crop_size)
        self.resize      = transforms.Resize((crop_size, crop_size), antialias=True)
        self.global_prob = global_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.global_prob:
            return self.resize(x)   # Full-frame global view
        return self.crop(x)         # Local detail crop

    def __repr__(self):
        return (f"RandomGlobalOrLocal(p_global={self.global_prob}, "
                f"crop={self.crop.size}, global_prob={self.global_prob})")


class ViratDataset(Dataset):
    def __init__(self, data_root, sequence_len=2, transform=None, video_folders=None, max_samples=None):
        """
        Args:
            data_root (str): Path to directory with processed_frames/video_name/frame_*.jpg.
            sequence_len (int): Number of consecutive frames to return.
            transform (callable, optional): Optional transform to be applied on a sample.
            video_folders (list, optional): List of folder paths to use. If None, uses all in data_root.
            max_samples (int, optional): If set, randomly subsample this many sequences from the full index.
        """
        self.data_root = data_root
        self.sequence_len = sequence_len
        self.transform = transform
        
        if video_folders is None:
            # Identify all video folders if not provided
            self.video_folders = [f.path for f in os.scandir(data_root) if f.is_dir()]
        else:
            self.video_folders = video_folders
            
        print(f"Dataset found {len(self.video_folders)} extracted videos.")
        
        # Build a flat index of all valid (video_idx, start_frame_idx) pairs
        self.video_files = []
        self.index_map = []  # List of (video_idx, start_frame_idx)
        for v_folder in self.video_folders:
            frames = natsorted(glob.glob(os.path.join(v_folder, "*.jpg")))
            if len(frames) >= self.sequence_len:
                v_idx = len(self.video_files)
                self.video_files.append(frames)
                for start in range(len(frames) - self.sequence_len + 1):
                    self.index_map.append((v_idx, start))

        # Randomly subsample if max_samples is set
        if max_samples is not None and max_samples < len(self.index_map):
            self.index_map = random.sample(self.index_map, max_samples)

        print(f"Dataset: {len(self.video_folders)} videos, {len(self.index_map)} sequences")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # Deterministic index lookup using the pre-built map
        video_idx, start_idx = self.index_map[idx]
        video_frames = self.video_files[video_idx]
        frame_paths = video_frames[start_idx : start_idx + self.sequence_len]
        
        frames = []
        for f_path in frame_paths:
            img = cv2.imread(f_path)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        # Convert to float tensors [0, 1]
        # (3, H, W)
        tensors = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
        
        if self.transform:
            # Apply same transform to all frames in the sequence (crucial for random crops)
            # To ensure consistent cropping across T, we combine them
            stacked = torch.stack(tensors) # (T, C, H, W)
            stacked = self.transform(stacked)
            return stacked
            
        return torch.stack(tensors)


def get_dataloaders(
    train_max_samples: int,
    val_max_samples: int,
    batch_size: int,
    data_path: str = DATA_PATH,
    sequence_len: int = 10,
    global_prob: float = 0.20,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
):
    all_folders = [f.path for f in os.scandir(data_path) if f.is_dir()]
    random.seed(42)
    random.shuffle(all_folders)

    n = len(all_folders)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = train_end + int(n * VAL_SPLIT)

    train_folders = all_folders[:train_end]
    val_folders   = all_folders[train_end:val_end]
    # test_folders  = all_folders[val_end:]  # Reserved for final evaluation

    print(f"Split: {len(train_folders)} train | {len(val_folders)} val | {n - val_end} test videos")

    # RandomGlobalOrLocal: 80% local crop (detail), 20% global resize (scene layout)
    transform = RandomGlobalOrLocal(crop_size=352, global_prob=global_prob)

    train_dataset = ViratDataset(DATA_PATH, sequence_len=sequence_len, transform=transform, video_folders=train_folders, max_samples=train_max_samples)
    val_dataset   = ViratDataset(DATA_PATH, sequence_len=sequence_len, transform=transform, video_folders=val_folders,   max_samples=val_max_samples)

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    processed_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed_frames')
    if os.path.exists(processed_path):
        dataset = ViratDataset(processed_path)
        sample = dataset[0]
        print(f"Sample shape (T, C, H, W): {sample.shape}")
