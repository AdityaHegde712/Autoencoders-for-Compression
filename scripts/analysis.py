"""
analysis.py
Performs deep analysis of the compression efficiency and latent space.
  1. Size Analysis: Raw vs. Compressed (Bytes).
  2. Latent PCA: Dimensionality reduction (128 -> 2) to study latent distribution.

Usage:
    python scripts/analysis.py --run ml/models/saved/<RUN_NAME>
"""
import argparse
import os
import sys
import math
import random

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import ViratDataset
from ml.models.autoencoder import AsymmetricAutoencoder
from ml.utils.device import get_device

# ── reproduce the data split ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(PROJECT_ROOT, 'data', 'processed_frames')
TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.15
IFRAME_PROB = 0.10
DEVICE      = get_device()


def get_test_folders():
    all_folders = [f.path for f in os.scandir(DATA_PATH) if f.is_dir()]
    random.seed(42)
    random.shuffle(all_folders)
    n         = len(all_folders)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = train_end + int(n * VAL_SPLIT)
    return all_folders[val_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to the run folder (contains best_model.pth)")
    parser.add_argument("--num_latent_samples", type=int, default=5000, help="Number of latent spatial vectors to sample for PCA")
    args = parser.parse_args()

    model_path = os.path.join(args.run, "best_model.pth")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # ── load model ──────────────────────────────────────────────────────────
    model = AsymmetricAutoencoder(in_channels=3, latent_channels=64).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {model_path}")

    # ── test dataset ────────────────────────────────────────────────────────
    test_folders = get_test_folders()
    dataset      = ViratDataset(DATA_PATH, sequence_len=2, video_folders=test_folders)
    loader       = DataLoader(dataset, batch_size=4, shuffle=True)  # Using small batch for variety
    print(f"Test split: {len(test_folders)} videos, {len(dataset)} sequences")

    latent_vectors = []
    stats = []

    with torch.no_grad():
        for i, frames in enumerate(tqdm(loader, desc="Collecting Analysis Data")):
            if i > 50: break  # Limit to 200 sequences for analysis speed
            
            frames = frames.to(DEVICE)
            f_prev, f_curr = frames[:, 0], frames[:, 1]

            is_iframe = random.random() < IFRAME_PROB
            target    = f_curr if is_iframe else (f_curr - f_prev)

            # Get latents and likelihoods
            # We assume the model's forward or an internal method provides y
            # Since forward returns (rec, p_y, y), we'll unpack all three.
            res_hat, p_y, y = model(target, training=False)

            # ── Size Analysis ───────────────────────────────────────────────
            # Total bits = -sum(log2(p(y)))
            total_bits = -torch.sum(torch.log2(p_y + 1e-6)).item()
            total_bytes = math.ceil(total_bits / 8)
            
            # Input size in bytes: Width * Height * 3 channels * 1 byte/chan
            # Note: Our input tensor is normalized to [0,1], but original is 8-bit.
            raw_bytes = target.shape[2] * target.shape[3] * 3
            
            for b in range(target.shape[0]):
                stats.append({
                    "type": "I-frame" if is_iframe else "P-frame",
                    "raw_bytes": raw_bytes,
                    "compressed_bytes": total_bytes,
                })

            # ── Latent Collection ───────────────────────────────────────────
            # Sample random spatial locations from the y tensor (B, 64, H', W')
            # y shape: (B, 64, H/8, W/8)
            y_flat = y.permute(0, 2, 3, 1).reshape(-1, 64).cpu().numpy()
            sampled_idx = np.random.choice(len(y_flat), min(100, len(y_flat)), replace=False)
            latent_vectors.append(y_flat[sampled_idx])

    # ── Process Compression Stats ───────────────────────────────────────────
    df = pd.DataFrame(stats)
    df["compression_ratio"] = df["raw_bytes"] / df["compressed_bytes"]
    
    summary = df.groupby("type")[["raw_bytes", "compressed_bytes", "compression_ratio"]].mean()
    
    print("\n" + "="*50)
    print("AVERAGE COMPRESSION PERFORMANCE (per frame)")
    print("="*50)
    print(summary.to_string())
    print("-" * 50)
    print(f"Total sequences analysed: {len(df)}")
    print("="*50)

    # ── Process Latent PCA ──────────────────────────────────────────────────
    X_latents = np.concatenate(latent_vectors, axis=0)
    # Subsample to the user's requested count for the plot
    if len(X_latents) > args.num_latent_samples:
        X_latents = X_latents[np.random.choice(len(X_latents), args.num_latent_samples, replace=False)]

    print(f"Performing PCA on {len(X_latents)} latent samples...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_latents)

    # Calculate magnitude for color-coding (represents "activity" of that feature)
    magnitudes = np.linalg.norm(X_latents, axis=1)

    # ── 2D Visualization ──────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=magnitudes, cmap='viridis', alpha=0.5, s=2)
    plt.colorbar(scatter, label="Latent Vector Magnitude (L2)")
    plt.title(f"Latent Space PCA (2D)\nExplained Variance: {sum(pca.explained_variance_ratio_[:2]):.2%}", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)

    out_pca_2d = os.path.join(args.run, "latent_space_pca_2D.png")
    plt.savefig(out_pca_2d, dpi=150, bbox_inches="tight")
    print(f"2D Visualization saved to: {out_pca_2d}")

    # ── 3D Visualization ──────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=magnitudes, cmap='viridis', alpha=0.4, s=2)
    
    colorbar = plt.colorbar(scatter, ax=ax, label="Latent Vector Magnitude (L2)", pad=0.1)
    ax.set_title(f"Latent Space PCA (3D)\nTotal Explained Variance: {sum(pca.explained_variance_ratio_):.2%}", fontsize=14)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    
    # Set a nice viewing angle
    ax.view_init(elev=20, azim=45)

    out_pca_3d = os.path.join(args.run, "latent_space_pca_3D.png")
    plt.savefig(out_pca_3d, dpi=150, bbox_inches="tight")
    print(f"3D Visualization saved to: {out_pca_3d}")

    # ── Save stats to CSV ───────────────────────────────────────────────────
    out_csv = os.path.join(args.run, "compression_analysis.csv")
    df.to_csv(out_csv, index=False)
    print(f"Full analysis data saved to: {out_csv}")


if __name__ == "__main__":
    main()
