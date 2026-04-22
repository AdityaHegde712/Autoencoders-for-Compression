"""
evaluate_compression.py

Benchmark aligned with demo/producer.py plus full decode path:
1) load model (optional random weights for throughput-only tests)
2) BGR image → resize (INTER_LINEAR) → preprocess → pad_to_multiple
3) encoder → convert_to_bitstream(latent)   # producer-style on encoder output
4) bottleneck → decoder → reconstruction (same order as model.forward decode path)

Timing (ms, averaged over --runs): preprocess, padding, encode, bitstream, bottleneck, decode, total per run.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.autoencoder import AsymmetricAutoencoder
from ml.utils.device import get_device, maybe_compile, synchronize
from demo.latent_bitstream import convert_to_bitstream
from demo.preprocess_gpu import preprocess_gpu
from demo.kafka_msg_processer import compress_for_kafka

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "data", "1080p-AHD-CCTV.jpg")


def randomize_model_weights(model: torch.nn.Module) -> None:
    """Fill model parameters with random values."""
    with torch.inference_mode():
        for param in model.parameters():
            param.copy_(torch.randn_like(param))


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 4):
    """Pad H and W to the nearest multiple (encoder downsamples by 4x)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def preprocess(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 HWC → RGB float32 NCHW tensor on device, values in [0, 1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().mul_(1.0 / 255.0)
    return tensor.to(device, non_blocking=device.type == "cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to image in data/")
    parser.add_argument("--latent-channels", type=int, default=64, help="Model latent channel count")
    parser.add_argument("--width", type=int, default=620, help="Resize width (producer default 620)")
    parser.add_argument("--height", type=int, default=480, help="Resize height (producer default 480)")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to .pth; if set, loads weights instead of randomizing",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    device = get_device()

    t0 = time.perf_counter()
    model = AsymmetricAutoencoder(in_channels=3, latent_channels=args.latent_channels, legacy_encoder=False).to(device)
    model.eval()
    encoder = maybe_compile(model.encoder, device)
    decoder = maybe_compile(model.decoder, device)
    t1 = time.perf_counter()

    if args.checkpoint:
        ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(PROJECT_ROOT, args.checkpoint)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        randomize_model_weights(model)
    t2 = time.perf_counter()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    pre_ms_total = 0.0
    pad_ms_total = 0.0
    enc_ms_total = 0.0
    bitstream_ms_total = 0.0
    bn_ms_total = 0.0
    dec_ms_total = 0.0
    run_total_ms = 0.0
    latent_bytes_total = 0
    bitstream_bytes_total = 0

    tensor = None
    latent = None
    bitstream = b""
    x_hat = None
    p_y = None

    with torch.inference_mode():
        for _ in range(args.runs):
            synchronize(device)
            t_run_start = time.perf_counter()

            t_pre_start = time.perf_counter()
            frame = cv2.resize(image_bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            tensor = preprocess_gpu(frame, device)
            synchronize(device)
            t_pre_end = time.perf_counter()
            pre_ms_total += (t_pre_end - t_pre_start) * 1000

            t_pad_start = time.perf_counter()
            target_padded, _, _ = pad_to_multiple(tensor, multiple=4)
            synchronize(device)
            t_pad_end = time.perf_counter()
            pad_ms_total += (t_pad_end - t_pad_start) * 1000

            t_enc_start = time.perf_counter()
            latent = encoder(target_padded)
            synchronize(device)
            t_enc_end = time.perf_counter()
            enc_ms_total += (t_enc_end - t_enc_start) * 1000

            t_bs_start = time.perf_counter()
            bitstream = convert_to_bitstream(latent)
            synchronize(device)
            t_bs_end = time.perf_counter()
            bitstream_ms_total += (t_bs_end - t_bs_start) * 1000
            latent_bytes_total += latent.numel() * latent.element_size()
            bitstream_bytes_total += len(bitstream)

            t_bn_start = time.perf_counter()
            y_q, p_y = model.bottleneck(latent, training=False)
            synchronize(device)
            t_bn_end = time.perf_counter()
            bn_ms_total += (t_bn_end - t_bn_start) * 1000

            t_dec_start = time.perf_counter()
            x_hat = decoder(y_q)
            synchronize(device)
            t_dec_end = time.perf_counter()
            dec_ms_total += (t_dec_end - t_dec_start) * 1000

            run_total_ms += (t_dec_end - t_run_start) * 1000

    t4 = time.perf_counter()

    n = args.runs
    print(f"Device               : {device}")
    print(f"Image                : {args.image}")
    print(f"Resize               : {args.width}x{args.height} (producer-style)")
    print(f"Timed runs           : {n}")
    print(f"Weights              : {'checkpoint ' + args.checkpoint if args.checkpoint else 'random'}")
    if tensor is not None:
        print(f"Input tensor shape   : {tuple(tensor.shape)} (after preprocess)")
    if latent is not None:
        print(f"Encoder latent shape : {tuple(latent.shape)}")
    if x_hat is not None:
        print(f"Decoded output shape : {tuple(x_hat.shape)}")
    if p_y is not None:
        print(f"Likelihood p_y shape   : {tuple(p_y.shape)}")
    print()
    print("Timing (ms, avg per run) — encode + bitstream + decode")
    print(f"Model init/load      : {(t1 - t0) * 1000:.6f}")
    print(f"Weights load/init    : {(t2 - t1) * 1000:.6f}")
    print(f"Preprocess           : {pre_ms_total / n:.6f}")
    print(f"Padding              : {pad_ms_total / n:.6f}")
    print(f"Encode (encoder)     : {enc_ms_total / n:.6f}")
    print(f"Bitstream conversion : {bitstream_ms_total / n:.6f}")
    print(f"Bottleneck           : {bn_ms_total / n:.6f}")
    print(f"Decode               : {dec_ms_total / n:.6f}")
    print(f"Pipeline total/run   : {run_total_ms / n:.6f}")
    print(f"Wall total           : {(t4 - t0) * 1000:.6f}")
    print()
    avg_latent_bytes = latent_bytes_total / n
    avg_bitstream_bytes = bitstream_bytes_total / n
    denom_pixels = tensor.shape[2] * tensor.shape[3] * 3 if tensor is not None else 1
    avg_bitstream_bpp = (avg_bitstream_bytes * 8) / denom_pixels
    print(f"Input image (resized): {frame.nbytes:.2f} bytes (per run)")
    if tensor is not None:
        print(f"Input tensor size    : {tensor.nbytes:.2f} bytes (per run)")
    print(f"Latent size          : {avg_latent_bytes:.2f} bytes (avg per run)")
    print(f"Decoder output size   : {x_hat.shape[2]}x{x_hat.shape[3]}x3  (avg per run)")
    print(f"Bitstream size       : {avg_bitstream_bytes:.2f} bytes (avg per run)")
    print(f"Bitstream BPP        : {avg_bitstream_bpp:.4f} (vs resized RGB)")


if __name__ == "__main__":
    main()
