"""
Consumer for demo/producer_dual_model.py: decode I-frames and cropped residuals
with iframe decoder, then restore full residual and reconstruct frames.
"""
import cv2
import torch
import numpy as np
import sys
import time
import blosc
import struct

from confluent_kafka import Consumer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.models.iframe_encoder import AsymmetricAutoencoder as IFrameAutoencoder
from ml.residual_extractor import restore_residual_from_crop
from ml.utils.device import get_device, maybe_compile
from demo.checkpoint_utils import strip_torch_compile_prefix
from demo.image_processing_gpu import postprocess_gpu

KAFKA_BROKER = "localhost:9092"
TOPIC = "ai-compressed-video"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IFRAME_CHECKPOINT = PROJECT_ROOT / "ml/models/saved/iframe_epoch_56_run1.pth"
DEVICE = get_device()

RESOLUTION_W = 1280
RESOLUTION_H = 720

IFRAME_LEGACY_ENCODER = False
GOP_SIZE = 5


def bitstream_to_tensor(bitstream, shape):
    decompressed_bytes = blosc.decompress(bitstream)
    flat_data = np.frombuffer(decompressed_bytes, dtype=np.float32)
    return torch.from_numpy(flat_data).reshape(shape)


def unpackage_frame(packet):
    header_size = struct.calcsize(">I?IIIIIIIIII")
    header_data = packet[:header_size]
    bitstream = packet[header_size:]
    (
        frame_idx,
        is_iframe,
        c,
        h,
        w,
        full_h,
        full_w,
        top,
        left,
        bottom,
        right,
        payload_len,
    ) = struct.unpack(">I?IIIIIIIIII", header_data)
    return {
        "idx": frame_idx,
        "is_iframe": is_iframe,
        "shape": (1, c, h, w),
        "full_shape": (3, full_h, full_w),
        "coords": (top, left, bottom, right),
        "bitstream": bitstream,
    }


iframe_model = IFrameAutoencoder(
    in_channels=3, latent_channels=64, legacy_encoder=IFRAME_LEGACY_ENCODER
).to(DEVICE)
iframe_model.load_state_dict(
    strip_torch_compile_prefix(
        torch.load(IFRAME_CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
)
iframe_model.eval()
iframe_decoder = maybe_compile(iframe_model.decoder, DEVICE)

c = Consumer(
    {
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "ai-viewer-group-dual",
        "auto.offset.reset": "latest",
    }
)
c.subscribe([TOPIC])

print(
    "Dual consumer started (GOP=%d). I-frame encoder checkpoint: %s"
    % (GOP_SIZE, IFRAME_CHECKPOINT)
)

try:
    with torch.inference_mode():
        frame_count = 0
        prev_hat = None
        iteration_times = []
        deser_times = []
        decode_times = []
        postproc_times = []

        while True:
            msg = c.poll(0.1)
            if msg is None:
                continue

            package = unpackage_frame(msg.value())
            is_iframe = bool(package["is_iframe"])
            if not is_iframe and prev_hat is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            iteration_start = time.time()

            deser_start = time.time()
            latent_q = bitstream_to_tensor(package["bitstream"], package["shape"]).to(
                DEVICE, non_blocking=DEVICE.type == "cuda"
            )
            deser_time = time.time() - deser_start

            full_shape = package["full_shape"]
            coords = package["coords"]
            h, w = full_shape[1], full_shape[2]

            decode_start = time.time()
            if is_iframe:
                x_hat = iframe_decoder(latent_q)
                prev_hat = x_hat
            else:
                cropped_res_hat = iframe_decoder(latent_q)
                full_res = restore_residual_from_crop(
                    cropped_res_hat.squeeze(0), coords, full_shape
                ).unsqueeze(0)
                prev_hat = (prev_hat + full_res).clamp(0, 1)
                x_hat = full_res
            x_hat = x_hat[:, :, :h, :w]
            decode_time = time.time() - decode_start

            postproc_start = time.time()
            reconstructed_frame = postprocess_gpu(x_hat)
            postproc_time = time.time() - postproc_start

            cv2.imshow("AI Decompressed Feed (dual)", reconstructed_frame)

            iteration_time = time.time() - iteration_start
            frame_count += 1
            iteration_times.append(iteration_time)
            deser_times.append(deser_time)
            decode_times.append(decode_time)
            postproc_times.append(postproc_time)

            if frame_count % GOP_SIZE == 0:
                n = len(iteration_times)
                avg_iter = (sum(iteration_times) / n) * 1000
                avg_deser = (sum(deser_times) / n) * 1000
                avg_decode = (sum(decode_times) / n) * 1000
                avg_postproc = (sum(postproc_times) / n) * 1000
                print(f"\n--- Last {GOP_SIZE} frames avg ---")
                print(f"  Deserialization: {avg_deser:.2f}ms")
                print(f"  Decoding: {avg_decode:.2f}ms")
                print(f"  Postprocessing: {avg_postproc:.2f}ms")
                print(f"  Total Iteration: {avg_iter:.2f}ms\n")
                iteration_times = []
                deser_times = []
                decode_times = []
                postproc_times = []

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    c.close()
    cv2.destroyAllWindows()
