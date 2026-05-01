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
from pytorch_msssim import ssim as calc_ssim, ms_ssim as calc_msssim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import ViratDataset
from ml.models.autoencoder import AsymmetricAutoencoder, LegacyAsymmetricAutoencoder
from ml.utils.compression_metrics import compression_ratio_from_bpp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(PROJECT_ROOT, 'data', 'processed_frames')
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


def infer_latent_channels(state_dict) -> int:
    return state_dict["bottleneck.log_scale"].shape[1]


def load_autoencoder_for_checkpoint(model_path: str, device: torch.device):
    state_dict = torch.load(model_path, map_location=device)
    latent_channels = infer_latent_channels(state_dict)
    uses_legacy_encoder = "encoder.0.weight" in state_dict
    model_cls = LegacyAsymmetricAutoencoder if uses_legacy_encoder else AsymmetricAutoencoder
    model = model_cls(in_channels=3, latent_channels=latent_channels).to(device)
    model.load_state_dict(state_dict)
    return model, latent_channels, model_cls.__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to the run folder (contains best_model.pth)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of test sequences to evaluate. Useful for quick smoke tests.",
    )
    args = parser.parse_args()

    model_path = os.path.join(args.run, "best_model.pth")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # ── load model ──────────────────────────────────────────────────────────
    model, latent_channels, model_name = load_autoencoder_for_checkpoint(model_path, DEVICE)
    model.eval()
    print(f"Loaded {model_name} from {model_path} (latent_channels={latent_channels})")

    # ── test dataset ────────────────────────────────────────────────────────
    test_folders = get_test_folders()
    dataset      = ViratDataset(DATA_PATH, sequence_len=2, video_folders=test_folders, max_samples=args.max_samples)
    loader       = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set: {len(test_folders)} videos, {len(dataset)} sequences, {len(loader)} batches")

    total_psnr, total_ssim, total_msssim, total_bpp = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for frames in tqdm(loader, desc="Evaluating"):
            frames = frames.to(DEVICE)
            f_prev, f_curr = frames[:, 0], frames[:, 1]

            is_iframe = random.random() < IFRAME_PROB
            target    = f_curr if is_iframe else (f_curr - f_prev)

            res_hat, p_y, _ = model(target, training=False)

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
            n_batches  += 1

    avg_psnr = total_psnr / n_batches
    avg_ssim = total_ssim / n_batches
    avg_msssim = total_msssim / n_batches
    avg_bpp  = total_bpp  / n_batches
    avg_compression_ratio = compression_ratio_from_bpp(avg_bpp)

    print("\n" + "=" * 40)
    print(f"  PSNR :  {avg_psnr:.2f} dB")
    print(f"  SSIM :  {avg_ssim:.4f}")
    print(f"  MS-SSIM :  {avg_msssim:.4f}")
    print(f"  BPP  :  {avg_bpp:.4f}")
    print(f"  Compression ratio vs raw 24-bit video :  {avg_compression_ratio:.2f}:1")
    print("=" * 40)

    # Save to results txt alongside the model
    out_path = os.path.join(args.run, "eval_results.txt")
    with open(out_path, "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.6f}\n")
        f.write(f"MS-SSIM: {avg_msssim:.6f}\n")
        f.write(f"BPP:  {avg_bpp:.6f}\n")
        f.write(f"Compression ratio vs raw 24-bit video: {avg_compression_ratio:.6f}:1\n")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
