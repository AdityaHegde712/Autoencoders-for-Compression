"""
visualize.py
Loads best_model.pth from a training run and renders a grid of:
  Original Frame | Reconstructed Frame | Error Map (5x magnified)
for a handful of randomly sampled test-split sequences.

Usage:
    python scripts/visualize.py --run ml/models/saved/<RUN_NAME>
    python scripts/visualize.py --run ml/models/saved/<RUN_NAME> --n 8
"""
import argparse
import os
import sys
import random

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import ViratDataset
from ml.models.autoencoder import AsymmetricAutoencoder

# ── replicate the train/val/test split ──────────────────────────────────────
DATA_PATH   = r'c:\Users\hifia\Projects\Autoencoders-for-Compression\data\processed_frames'
TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.15
IFRAME_PROB = 0.10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_folders():
    all_folders = [f.path for f in os.scandir(DATA_PATH) if f.is_dir()]
    random.seed(42)
    random.shuffle(all_folders)
    n         = len(all_folders)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = train_end + int(n * VAL_SPLIT)
    return all_folders[val_end:]


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [0,1] to uint8 numpy (H, W, C) RGB."""
    return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to the run folder (contains best_model.pth)")
    parser.add_argument("--n",   type=int, default=6, help="Number of frame pairs to visualise")
    args = parser.parse_args()

    model_path = os.path.join(args.run, "best_model.pth")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # ── load model ──────────────────────────────────────────────────────────
    model = AsymmetricAutoencoder(in_channels=3, latent_channels=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded model from {model_path}")
    print(f"Parameters: {total_params:,} total  |  {trainable_params:,} trainable")

    # ── test dataset ────────────────────────────────────────────────────────
    test_folders = get_test_folders()
    dataset      = ViratDataset(DATA_PATH, sequence_len=2, video_folders=test_folders)
    indices      = random.sample(range(len(dataset)), min(args.n, len(dataset)))

    rows = []
    with torch.no_grad():
        for idx in indices:
            frames = dataset[idx]                                # (2, C, H, W)
            f_prev = frames[0].unsqueeze(0).to(DEVICE)          # (1, C, H, W)
            f_curr = frames[1].unsqueeze(0).to(DEVICE)

            is_iframe = random.random() < IFRAME_PROB
            target    = f_curr if is_iframe else (f_curr - f_prev)

            res_hat, _, _ = model(target, training=False)

            # Reconstruct full frame
            if is_iframe:
                f_hat = res_hat.clamp(0, 1)
            else:
                f_hat = (f_prev + res_hat).clamp(0, 1)

            orig_img  = tensor_to_img(f_curr[0])
            recon_img = tensor_to_img(f_hat[0])
            error_img = np.abs(orig_img.astype(np.int16) - recon_img.astype(np.int16))
            error_img = (error_img * 5).clip(0, 255).astype(np.uint8)

            rows.append((orig_img, recon_img, error_img, "I" if is_iframe else "P"))

    # ── plot ────────────────────────────────────────────────────────────────
    n    = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Original", "Reconstructed", "Error ×5"]
    for row_idx, (orig, recon, err, ftype) in enumerate(rows):
        for col_idx, (img, title) in enumerate(zip([orig, recon, err], col_titles)):
            ax = axes[row_idx][col_idx]
            ax.imshow(img)
            label = f"{title}" if row_idx > 0 else f"{title} ({ftype}-frame)"
            ax.set_title(label, fontsize=10)
            ax.axis("off")

    plt.suptitle("Autoencoder Compression — Visual Inspection", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(args.run, "visual_inspection.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
