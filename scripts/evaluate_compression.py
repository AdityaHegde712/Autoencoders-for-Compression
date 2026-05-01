"""
evaluate_compression.py

Simple timing test for the autoencoder:
1) load/create model
2) fill model weights with random values
3) load and preprocess a photo from data/
4) run one forward pass
"""
import argparse
import os
import sys
import time

import cv2
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.autoencoder import AsymmetricAutoencoder
from ml.utils.compression_metrics import (
    compression_ratio_against_raw_video,
    raw_video_size_bytes,
)
from demo.compression import convert_to_bitstream


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "data", "1080p-AHD-CCTV.jpg")


def randomize_model_weights(model: torch.nn.Module) -> None:
    """Fill model parameters with random values."""
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param))


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to image in data/")
    parser.add_argument("--latent-channels", type=int, default=32, help="Model latent channel count")
    parser.add_argument("--height", type=int, default=1920, help="Resize height before inference")
    parser.add_argument("--width", type=int, default=1080, help="Resize width before inference")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    model = AsymmetricAutoencoder(in_channels=3, latent_channels=args.latent_channels).to(device)
    model.eval()
    t1 = time.perf_counter()

    randomize_model_weights(model)
    t2 = time.perf_counter()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    t3 = time.perf_counter()

    with torch.no_grad():
        sync_cuda(device)
        t_enc_start = time.perf_counter()
        y = model.encoder(x)
        y_q, p_y = model.bottleneck(y, training=False)
        sync_cuda(device)
        t_enc_end = time.perf_counter()

        t_dec_start = time.perf_counter()
        x_hat = model.decoder(y_q)
        sync_cuda(device)
        t_dec_end = time.perf_counter()

        t_bitstream_start = time.perf_counter()
        bitstream = convert_to_bitstream(y_q)
        sync_cuda(device)
        t_bitstream_end = time.perf_counter()
    t4 = t_dec_end

    print(f"Device               : {device}")
    print(f"Image                : {args.image}")
    print(f"Input tensor shape   : {tuple(x.shape)}")
    print(f"Output tensor shape  : {tuple(x_hat.shape)}")
    print(f"Likelihood shape     : {tuple(p_y.shape)}")
    print(f"Latent shape         : {tuple(y.shape)}")
    print()
    print("Timing (ms)")
    print(f"Model init/load      : {(t1 - t0) * 1000:.6f}")
    print(f"Randomize weights    : {(t2 - t1) * 1000:.6f}")
    print(f"Image preprocess     : {(t3 - t2) * 1000:.6f}")
    print(f"Encode + bottleneck  : {(t_enc_end - t_enc_start) * 1000:.6f}")
    print(f"Decode               : {(t_dec_end - t_dec_start) * 1000:.6f}")
    print(f"Bitstream conversion : {(t_bitstream_end - t_bitstream_start) * 1000:.6f}")
    print(f"Model pass total     : {(t4 - t3) * 1000:.6f}")
    print(f"Total                : {(t4 - t0) * 1000:.6f}")
    print()
    raw_video_bytes = raw_video_size_bytes(x.shape[2], x.shape[3])
    compression_ratio = compression_ratio_against_raw_video(
        len(bitstream),
        x.shape[2],
        x.shape[3],
    )

    print(f"Input image size    : {image.nbytes} bytes")
    print(f"Raw 24-bit frame    : {raw_video_bytes} bytes")
    print(f"Input tensor size   : {x.nbytes} bytes")
    print(f"Latent size         : {y_q.nbytes} bytes")
    print(f"Bitstream size      : {len(bitstream)} bytes")
    print(f"Bitstream BPP       : {len(bitstream)*8 / (x.shape[2] * x.shape[3]):.2f}")
    print(f"Compression ratio   : {compression_ratio:.2f}:1 vs raw 24-bit video")



if __name__ == "__main__":
    main()
