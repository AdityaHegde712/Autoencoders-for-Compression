"""
evaluate.py
Runs the model on the held-out test split and reports:
  - PSNR (dB)         ← standard compression quality benchmark
  - SSIM              ← perceptual quality
  - BPP               ← estimated bits-per-pixel (compression efficiency)

Usage:
    python scripts/evaluate.py --run ml/models/saved/<RUN_NAME>
"""
import argparse
import os
import sys
import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as calc_ssim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import ViratDataset
from ml.models.autoencoder import AsymmetricAutoencoder

DATA_PATH   = r'c:\Users\hifia\Projects\Autoencoders-for-Compression\data\processed_frames'
TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.15
IFRAME_PROB = 0.10
BATCH_SIZE  = 16
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_folders():
    all_folders = [f.path for f in os.scandir(DATA_PATH) if f.is_dir()]
    random.seed(42)
    random.shuffle(all_folders)
    n         = len(all_folders)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = train_end + int(n * VAL_SPLIT)
    return all_folders[val_end:]


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to the run folder (contains best_model.pth)")
    args = parser.parse_args()

    model_path = os.path.join(args.run, "best_model.pth")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # ── load model ──────────────────────────────────────────────────────────
    model = AsymmetricAutoencoder(in_channels=3, latent_channels=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {model_path}")

    # ── test dataset ────────────────────────────────────────────────────────
    test_folders = get_test_folders()
    dataset      = ViratDataset(DATA_PATH, sequence_len=2, video_folders=test_folders)
    loader       = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set: {len(test_folders)} videos, {len(dataset)} sequences, {len(loader)} batches")

    total_psnr, total_ssim, total_bpp = 0.0, 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for frames in tqdm(loader, desc="Evaluating"):
            frames = frames.to(DEVICE)
            f_prev, f_curr = frames[:, 0], frames[:, 1]

            is_iframe = random.random() < IFRAME_PROB
            target    = f_curr if is_iframe else (f_curr - f_prev)

            res_hat, p_y = model(target, training=False)

            # Reconstruct the full frame for quality metrics
            if is_iframe:
                f_hat = res_hat.clamp(0, 1)
            else:
                f_hat = (f_prev + res_hat).clamp(0, 1)

            # PSNR vs original full frame
            batch_psnr = psnr(f_hat, f_curr)

            # SSIM vs original full frame (full resolution)
            batch_ssim = calc_ssim(f_hat, f_curr, data_range=1.0, size_average=True).item()

            # BPP estimate
            num_pixels = f_curr.shape[0] * f_curr.shape[2] * f_curr.shape[3]
            batch_bpp  = (-torch.sum(torch.log2(p_y + 1e-6)) / num_pixels).item()

            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_bpp  += batch_bpp
            n_batches  += 1

    avg_psnr = total_psnr / n_batches
    avg_ssim = total_ssim / n_batches
    avg_bpp  = total_bpp  / n_batches

    print("\n" + "=" * 40)
    print(f"  PSNR :  {avg_psnr:.2f} dB")
    print(f"  SSIM :  {avg_ssim:.4f}")
    print(f"  BPP  :  {avg_bpp:.4f}")
    print("=" * 40)

    # Save to results txt alongside the model
    out_path = os.path.join(args.run, "eval_results.txt")
    with open(out_path, "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.6f}\n")
        f.write(f"BPP:  {avg_bpp:.6f}\n")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
