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
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as calc_ssim, ms_ssim as calc_msssim
from tqdm import tqdm

from ml.dataset import ViratDataset, get_test_folders
from ml.utils.constants import DATA_PATH, IFRAME_PROB, get_device
from ml.utils.model_loading import detect_latent_channels, load_model

BATCH_SIZE  = 8
TEST_MAX_SAMPLES = 2000
DEVICE      = get_device()


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
    print(f"\n[Step 1] Detecting and loading model...")
    detected_channels = detect_latent_channels(model_path)
    model = load_model(model_path, detected_channels, DEVICE)
    print(f"Loaded {type(model).__name__} with {detected_channels} channels from {model_path}")

    # ── test dataset ────────────────────────────────────────────────────────
    test_folders = get_test_folders()
    dataset      = ViratDataset(DATA_PATH, sequence_len=2, video_folders=test_folders, max_samples=TEST_MAX_SAMPLES)
    loader       = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set: {len(test_folders)} videos, {len(dataset)} sequences, {len(loader)} batches")

    total_psnr, total_ssim, total_bpp = 0.0, 0.0, 0.0
    total_comp_ms, total_decomp_ms = 0.0, 0.0
    n_batches = 0
    total_frames = 0

    with torch.no_grad():
        for frames in tqdm(loader, desc="Evaluating"):
            frames = frames.to(DEVICE)
            f_prev, f_curr = frames[:, 0], frames[:, 1]

            is_iframe = random.random() < IFRAME_PROB
            target    = f_curr if is_iframe else (f_curr - f_prev)

            # --- Measure Compression (Encoder + Bottleneck) ---
            comp_start = time.perf_counter()
            y = model.encoder(target)
            y_q, p_y = model.bottleneck(y, training=False)
            comp_end = time.perf_counter()

            # --- Measure Decompression (Decoder) ---
            decomp_start = time.perf_counter()
            res_hat = model.decoder(y_q)
            decomp_end = time.perf_counter()

            comp_ms = (comp_end - comp_start) * 1000
            decomp_ms = (decomp_end - decomp_start) * 1000

            # Reconstruct the full frame for quality metrics
            if is_iframe:
                f_hat = res_hat.clamp(0, 1)
            else:
                f_hat = (f_prev + res_hat).clamp(0, 1)

            # PSNR vs original full frame
            batch_psnr = psnr(f_hat, f_curr)

            # SSIM vs original full frame (full resolution)
            batch_ssim = calc_ssim(f_hat, f_curr, data_range=1.0, size_average=True).item()

            # MS-SSIM vs original full frame (full resolution)
            batch_msssim = calc_msssim(f_hat, f_curr, data_range=1.0, size_average=True).item()
            # BPP estimate
            num_pixels = f_curr.shape[0] * f_curr.shape[2] * f_curr.shape[3]
            batch_bpp  = (-torch.sum(torch.log2(p_y + 1e-6)) / num_pixels).item()

            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_msssim += batch_msssim
            total_bpp  += batch_bpp
            total_comp_ms += comp_ms
            total_decomp_ms += decomp_ms
            n_batches  += 1
            total_frames += f_curr.shape[0]

    avg_psnr = total_psnr / n_batches
    avg_ssim = total_ssim / n_batches
    avg_msssim = total_msssim / n_batches
    avg_bpp  = total_bpp  / n_batches
    avg_comp_ms = total_comp_ms / total_frames
    avg_decomp_ms = total_decomp_ms / total_frames

    print("\n" + "=" * 45)
    print(f"  QUALITY METRICS")
    print(f"  PSNR :  {avg_psnr:.2f} dB")
    print(f"  SSIM :  {avg_ssim:.4f}")
    print(f"  MS-SSIM :  {avg_msssim:.4f}")
    print(f"  BPP  :  {avg_bpp:.4f}")
    print("-" * 45)
    print(f"  PERFORMANCE (avg ms per frame)")
    print(f"  Compression:   {avg_comp_ms:.2f} ms")
    print(f"  Decompression: {avg_decomp_ms:.2f} ms")
    print(f"  Total Latency: {avg_comp_ms + avg_decomp_ms:.2f} ms")
    print("=" * 45)

    # Save to results txt alongside the model
    out_path = os.path.join(args.run, "eval_results.txt")
    with open(out_path, "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.6f}\n")
        f.write(f"MS-SSIM: {avg_msssim:.6f}\n")
        f.write(f"BPP:  {avg_bpp:.6f}\n")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
