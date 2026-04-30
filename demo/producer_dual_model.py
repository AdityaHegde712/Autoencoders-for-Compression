"""
Kafka producer: I-frames encoded with ml.models.iframe_encoder, P-frames with
ml.models.pframe_encoder, GOP=3. Pair with demo/consumer_dual_model.py.
"""
import cv2
import torch
import sys
import time
import struct

from confluent_kafka import Producer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.models.iframe_encoder import AsymmetricAutoencoder as IFrameAutoencoder
from ml.models.pframe_encoder import AsymmetricAutoencoder as PFrameAutoencoder
from ml.utils.device import get_device, maybe_compile
from demo.checkpoint_utils import strip_torch_compile_prefix
from demo.latent_bitstream import convert_to_bitstream
from demo.image_processing_gpu import preprocess_gpu
from demo.kafka_msg_processer import compress_frame_for_kafka
# --- CONFIG ---
KAFKA_BROKER = "localhost:9092"
TOPIC = "ai-compressed-video"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Set paths to checkpoints trained with the matching encoder modules / legacy flags below.
IFRAME_CHECKPOINT = PROJECT_ROOT / "ml/models/saved/iframe_epoch_56_run1.pth"
PFRAME_CHECKPOINT = PROJECT_ROOT / "ml/models/saved/pframe_epoch_12_run2.pth"

DEVICE = get_device()
RESOLUTION_W = 1280
RESOLUTION_H = 720
GOP_SIZE = 3

# Must match how each checkpoint was trained (see ml/train_iframe.py / ml/train_pframe.py).
IFRAME_LEGACY_ENCODER = False
PFRAME_LEGACY_ENCODER = False


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 4):
    """Pad H and W to the nearest multiple (encoder downsamples by 4x or 8x)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def package_frame(frame_idx, is_iframe, latent, bitstream):
    _, c, h, w = latent.shape
    header_format = ">I?IIII"
    header = struct.pack(header_format, frame_idx, is_iframe, c, h, w, len(bitstream))
    return header + bitstream


# --- Load I-frame and P-frame models (separate checkpoints) ---
iframe_model = IFrameAutoencoder(
    in_channels=3, latent_channels=64, legacy_encoder=IFRAME_LEGACY_ENCODER
).to(DEVICE)
pframe_model = PFrameAutoencoder(
    in_channels=3, latent_channels=64, legacy_encoder=PFRAME_LEGACY_ENCODER
).to(DEVICE)

iframe_model.load_state_dict(
    strip_torch_compile_prefix(
        torch.load(IFRAME_CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
)
pframe_model.load_state_dict(
    strip_torch_compile_prefix(
        torch.load(PFRAME_CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
)
iframe_model.eval()
pframe_model.eval()

iframe_encoder = maybe_compile(iframe_model.encoder, DEVICE)
pframe_encoder = maybe_compile(pframe_model.encoder, DEVICE)

p = Producer({"bootstrap.servers": KAFKA_BROKER, "message.max.bytes": 5000000})
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Producer started (dual model, GOP=%d). I-frame: %s | P-frame: %s" % (GOP_SIZE, IFRAME_CHECKPOINT, PFRAME_CHECKPOINT))

try:
    with torch.inference_mode():
        frame_idx = 0
        prev_hat = None
        gop_count = 0
        inter = 0

        preprocess_total_time = 0
        padding_total_time = 0
        neural_total_time = 0
        bitstream_total_time = 0
        iteration_total_time = 0

        ave_frame_size = 0
        ave_tensor_size = 0
        ave_raw_size = 0
        ave_com_size = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            inter = (inter + 1) % GOP_SIZE
            if inter == 0:
                preprocess_total_time = 0
                padding_total_time = 0
                neural_total_time = 0
                bitstream_total_time = 0
                iteration_total_time = 0
                ave_frame_size = 0
                ave_tensor_size = 0
                ave_raw_size = 0
                ave_com_size = 0
            iteration_start = time.time()

            preprocess_start = time.time()
            frame = cv2.resize(frame, (RESOLUTION_W, RESOLUTION_H), interpolation=cv2.INTER_LINEAR)
            tensor = preprocess_gpu(frame, DEVICE)
            preprocess_time = time.time() - preprocess_start
            preprocess_total_time += preprocess_time

            padding_start = time.time()
            target_padded, _, _ = pad_to_multiple(tensor, multiple=4)
            padding_time = time.time() - padding_start
            padding_total_time += padding_time

            frame_in_gop = frame_idx % GOP_SIZE
            is_iframe = frame_in_gop == 0

            neural_start = time.time()
            if is_iframe:
                inp = target_padded
                latent = iframe_encoder(inp)
                x_hat, _, _ = iframe_model(inp, training=False)
                prev_hat = x_hat
            else:
                residual = target_padded - prev_hat
                inp = residual
                latent = pframe_encoder(inp)
                res_hat, _, _ = pframe_model(inp, training=False)
                prev_hat = (prev_hat + res_hat).clamp(0, 1)
            neural_time = time.time() - neural_start
            neural_total_time += neural_time

            bitstream_start = time.time()
            latent_compressed = convert_to_bitstream(latent)
            bitstream_time = time.time() - bitstream_start
            bitstream_total_time += bitstream_time

            iteration_time = time.time() - iteration_start
            iteration_total_time += iteration_time

            ave_frame_size += frame.nbytes
            ave_tensor_size += tensor.nbytes
            ave_raw_size += latent.nbytes
            ave_com_size += len(latent_compressed)

            if inter == GOP_SIZE - 1:
                gop_count += 1
                print(f"\n--- GOP {gop_count} Timing ---")
                print(f"  Preprocess: {preprocess_total_time * (1000 / GOP_SIZE):.2f}ms")
                print(f"  Padding: {padding_total_time * (1000 / GOP_SIZE):.2f}ms")
                print(f"  Neural Pipeline: {neural_total_time * (1000 / GOP_SIZE):.2f}ms")
                print(f"  Bitstream Conversion: {bitstream_total_time * (1000 / GOP_SIZE):.2f}ms")
                print(f"  Total Iteration: {iteration_total_time * (1000 / GOP_SIZE):.2f}ms")
                print("--------")
                print(f"Original frame  size: {ave_frame_size / GOP_SIZE} bytes")
                print(f"Original tensor size: {ave_tensor_size / GOP_SIZE} bytes")
                print(f"Original latent size: {ave_raw_size / GOP_SIZE} bytes")
                print(f"Compressed bitstream: {ave_com_size / GOP_SIZE} bytes")
                print(f"Compressed frame    : {len(compress_frame_for_kafka(frame))} bytes")

            packet = package_frame(frame_idx, is_iframe, latent, latent_compressed)
            frame_idx += 1

            p.produce(TOPIC, value=packet)
            p.poll(0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    cv2.destroyAllWindows()

cap.release()
