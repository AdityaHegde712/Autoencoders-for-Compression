"""
Real-time camera compression demo.

Left: raw camera feed
Right: compressed + decompressed through the autoencoder
Overlay: encode / decode / total pipeline timing in ms

Usage:
    python scripts/live_demo.py
    python scripts/live_demo.py --checkpoint ml/models/saved/run_01/best_model.pth
    python scripts/live_demo.py --camera 1          # use a different camera index
    python scripts/live_demo.py --resolution 480    # vertical resolution (default 480)

Press 'q' to quit.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.models.autoencoder import AsymmetricAutoencoder
from ml.utils.device import get_device, synchronize

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "ml/models/saved/run_01/best_model.pth"
LATENT_CHANNELS = 64
WINDOW_NAME = "Autoencoder Live Demo  |  Press Q to quit"


def load_model(checkpoint_path: str, device: torch.device) -> AsymmetricAutoencoder:
    model = AsymmetricAutoencoder(in_channels=3, latent_channels=LATENT_CHANNELS).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.encoder = torch.compile(model.encoder)
    model.decoder = torch.compile(model.decoder)
    return model


def preprocess(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 HWC → RGB float32 NCHW tensor on device, values in [0, 1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """RGB float32 NCHW tensor → BGR uint8 HWC numpy array."""
    img = tensor.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 4):
    """Pad H and W to the nearest multiple (encoder downsamples by 4x)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def draw_stats(canvas, stats: dict, x_offset: int, y_offset: int = 30):
    """Draw timing stats on the canvas."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    color = (0, 255, 0)
    thickness = 1
    line_gap = 24

    lines = [
        f"Encode:  {stats['encode_ms']:.1f} ms",
        f"Decode:  {stats['decode_ms']:.1f} ms",
        f"Total:   {stats['total_ms']:.1f} ms",
        f"FPS:     {stats['fps']:.1f}",
    ]
    for i, line in enumerate(lines):
        pos = (x_offset + 10, y_offset + i * line_gap)
        # shadow for readability
        cv2.putText(canvas, line, (pos[0] + 1, pos[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, line, pos, font, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Real-time autoencoder compression demo")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--resolution", type=int, default=480, help="Vertical resolution to capture")
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────────────
    device = get_device()
    print(f"Using device: {device}")

    # ── model ───────────────────────────────────────────────────────────
    ckpt_path = Path(__file__).resolve().parent.parent / args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = load_model(str(ckpt_path), device)
    print(f"Model loaded from {ckpt_path}")

    # ── camera ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.resolution * 4 / 3))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution)

    # exponential moving average for smoother stats display
    ema = {"encode_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0, "fps": 0.0}
    alpha = 0.3  # smoothing factor

    print("Starting live demo — press 'q' in the window to quit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # resize to requested resolution keeping aspect ratio
            h0, w0 = frame.shape[:2]
            scale = args.resolution / h0
            new_w = int(w0 * scale)
            frame = cv2.resize(frame, (new_w, args.resolution))
            frame = cv2.flip(frame, 1)

            # ── encode ──────────────────────────────────────────────
            t0 = time.perf_counter()
            tensor = preprocess(frame, device)
            tensor_padded, orig_h, orig_w = pad_to_multiple(tensor, multiple=4)
            y = model.encoder(tensor_padded)
            y_q, _ = model.bottleneck(y, training=False)
            synchronize(device)
            t_encode = time.perf_counter()

            # ── decode ──────────────────────────────────────────────
            x_hat = model.decoder(y_q)
            # crop padding back
            x_hat = x_hat[:, :, :orig_h, :orig_w]
            synchronize(device)
            t_decode = time.perf_counter()

            reconstructed = postprocess(x_hat)

            # ── timing ──────────────────────────────────────────────
            encode_ms = (t_encode - t0) * 1000
            decode_ms = (t_decode - t_encode) * 1000
            total_ms = (t_decode - t0) * 1000
            fps = 1000.0 / total_ms if total_ms > 0 else 0

            for k, v in [("encode_ms", encode_ms), ("decode_ms", decode_ms),
                          ("total_ms", total_ms), ("fps", fps)]:
                ema[k] = alpha * v + (1 - alpha) * ema[k]

            # ── build side-by-side canvas ───────────────────────────
            h, w = frame.shape[:2]
            divider = np.full((h, 2, 3), 128, dtype=np.uint8)
            canvas = np.hstack([frame, divider, reconstructed])

            # labels
            cv2.putText(canvas, "Original", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Reconstructed", (w + 12, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            draw_stats(canvas, ema, x_offset=w + 2, y_offset=25)

            cv2.imshow(WINDOW_NAME, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
