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
import random

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA

from ml.dataset import ViratDataset, get_test_folders
from ml.utils.constants import DATA_PATH, IFRAME_PROB, get_device
from ml.utils.model_loading import detect_latent_channels, load_model

DEVICE      = get_device()


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [0,1] to uint8 numpy (H, W, C) RGB."""
    return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to the run folder (contains best_model.pth)")
    parser.add_argument("--n",   type=int, default=6, help="Number of frame pairs to visualise")
    parser.add_argument("--num-latent-samples", type=int, default=5000, help="Number of latent spatial vectors to sample for PCA")
    args = parser.parse_args()

    model_path = os.path.join(args.run, "best_model.pth")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # ── load model ──────────────────────────────────────────────────────────
    print(f"\n[Step 1] Detecting and loading model...")
    detected_channels = detect_latent_channels(model_path)
    model = load_model(model_path, detected_channels, DEVICE)

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

    # ── Latent Collection (full test set via DataLoader) ─────────────────────
    print("\n[Step 2] Collecting latent vectors from test set...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    latent_vectors = []
    with torch.no_grad():
        for i, frames in enumerate(tqdm(loader, desc="Collecting Latents")):
            if i > 50:
                break  # Limit number of batches for speed
            frames = frames.to(DEVICE)
            f_prev, f_curr = frames[:, 0], frames[:, 1]

            is_iframe = random.random() < IFRAME_PROB
            target = f_curr if is_iframe else (f_curr - f_prev)

            _, _, y = model(target, training=False)

            # y shape: (B, C, H', W') — flatten spatial dims
            y_flat = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1]).cpu().numpy()
            # Sample a deterministic subset for variety
            rng = np.random.RandomState(42)
            n_latent = min(100, len(y_flat))
            sampled_idx = rng.choice(len(y_flat), n_latent, replace=False)
            latent_vectors.append(y_flat[sampled_idx])

    # ── Latent Space PCA Visualization ───────────────────────────────────────
    print("\n[Step 3] Computing latent space PCA...")
    X_latents = np.concatenate(latent_vectors, axis=0)
    # Subsample to the requested count for the plot
    if len(X_latents) > args.num_latent_samples:
        rng = np.random.RandomState(42)
        X_latents = X_latents[rng.choice(len(X_latents), args.num_latent_samples, replace=False)]

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_latents)

    # Colour by latent vector magnitude (represents "activity" of that feature)
    magnitudes = np.linalg.norm(X_latents, axis=1)

    # ── 2D PCA ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=magnitudes, cmap='viridis', alpha=0.5, s=2)
    plt.colorbar(scatter, label="Latent Vector Magnitude (L2)")
    plt.title(f"Latent Space PCA (2D)\nExplained Variance: {sum(pca.explained_variance_ratio_[:2]):.2%}", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)

    out_pca_2d = os.path.join(args.run, "latent_space_pca_2D.png")
    plt.savefig(out_pca_2d, dpi=150, bbox_inches="tight")
    print(f"2D PCA saved: {out_pca_2d}")

    # ── 3D PCA ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                         c=magnitudes, cmap='viridis', alpha=0.4, s=2)
    plt.colorbar(scatter, ax=ax, label="Latent Vector Magnitude (L2)", pad=0.1)
    ax.set_title(f"Latent Space PCA (3D)\nTotal Explained Variance: {sum(pca.explained_variance_ratio_):.2%}", fontsize=14)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.view_init(elev=20, azim=45)

    out_pca_3d = os.path.join(args.run, "latent_space_pca_3D.png")
    plt.savefig(out_pca_3d, dpi=150, bbox_inches="tight")
    print(f"3D PCA saved: {out_pca_3d}")


if __name__ == "__main__":
    main()
