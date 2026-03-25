import cv2
import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from natsort import natsorted

# Ensure the root project directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

if __name__ == "__main__":
    # Test the dataset
    processed_path = r'c:\Users\hifia\Projects\Autoencoders-for-Compression\data\processed_frames'
    if os.path.exists(processed_path):
        dataset = ViratDataset(processed_path)
        sample = dataset[0]
        print(f"Sample shape (T, C, H, W): {sample.shape}")
