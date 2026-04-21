import cv2
import torch
import sys
import time 
import struct
import numpy as np

from confluent_kafka import Producer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.models.autoencoder_nrm_conv import AsymmetricAutoencoder as AsymmetricAutoencoder_nrm_conv
from ml.utils.device import get_device, maybe_compile
from demo.latent_bitstream import convert_to_bitstream

# --- CONFIG ---
KAFKA_BROKER = 'localhost:9092'
TOPIC = 'ai-compressed-video'
CHECKPOINT = "../ml/models/saved/best_model.pth"
DEVICE = get_device()

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


def package_frame(frame_idx, is_iframe, latent, bitstream):
    # 1. Get metadata
    _, c, h, w = latent.shape
    
    # 2. Define the header format: 
    # I = unsigned int (4 bytes), ? = bool (1 byte)
    # Format: FrameIdx (I), IsIFrame (?), C (I), H (I), W (I), PayloadLen (I)
    header_format = ">I?IIII" 
    header = struct.pack(header_format, frame_idx, is_iframe, c, h, w, len(bitstream))
    
    # 3. Combine header and the torchac bitstream
    packet = header + bitstream
    return packet

# 1. Load Model
model = AsymmetricAutoencoder_nrm_conv(in_channels=3, latent_channels=64).to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()
encoder = maybe_compile(model.encoder, DEVICE)

p = Producer({'bootstrap.servers': KAFKA_BROKER, 'message.max.bytes': 5000000})
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Producer started. Sending AI-compressed latent tensors...")

try:
    with torch.inference_mode():
        gop_size = 10
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
            if not ret: break
            
            # --- TIMING: Start iteration timer ---
            inter = inter + 1
            inter = inter % gop_size

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
            
            # Compression Pipeline
            # Preprocess -> Pad -> Encode -> Bottleneck
            
            # 1. Preprocess and buffer
            preprocess_start = time.time()
            frame = cv2.resize(frame, (620, 480), interpolation=cv2.INTER_LINEAR)
            tensor = preprocess(frame, DEVICE)
            preprocess_time = time.time() - preprocess_start
            preprocess_total_time += preprocess_time

            # Padding for the model (multiple of 4)
            padding_start = time.time()
            target_padded, orig_h, orig_w = pad_to_multiple(tensor, multiple=4)
            padding_time = time.time() - padding_start
            padding_total_time += padding_time

            #  Neural Pipeline
            neural_start = time.time()
            latent = encoder(target_padded)
            neural_time = time.time() - neural_start
            neural_total_time += neural_time
            
            # Convert to bitstream
            bitstream_start = time.time()
            latent_compressed = convert_to_bitstream(latent)
            bitstream_time = time.time() - bitstream_start
            bitstream_total_time += bitstream_time
            
            # --- TIMING: End iteration timer and display results ---
            iteration_time = time.time() - iteration_start
            iteration_total_time += iteration_time

            ave_frame_size += frame.nbytes
            ave_tensor_size += tensor.nbytes
            ave_raw_size += latent.nbytes
            ave_com_size += len(latent_compressed)
            
            #print ave metric
            if inter == gop_size - 1:
                gop_count += 1
                print(f"\n--- Frame {gop_count} Timing ---")
                print(f"  Preprocess: {preprocess_total_time*(1000/gop_size):.2f}ms")
                print(f"  Padding: {padding_total_time*(1000/gop_size):.2f}ms")
                print(f"  Neural Pipeline: {neural_total_time*(1000/gop_size):.2f}ms")
                print(f"  Bitstream Conversion: {bitstream_total_time*(1000/gop_size):.2f}ms")
                print(f"  Total Iteration: {iteration_total_time*(1000/gop_size):.2f}ms")
                print("--------")
                print(f"Original frame  size: {ave_frame_size/gop_count} bytes")
                print(f"Original tensor size: {ave_tensor_size/gop_count} bytes")
                print(f"Original latent size: {ave_raw_size/gop_count} bytes")
                print(f"Compressed bitstream: {ave_com_size/gop_count} bytes")
            
            #packet = package_frame(1, False, latent ,latent_compressed)

            #p.produce(TOPIC, value=packet)
            #p.poll(0) # Serve delivery callbacks

        

            

            """
            # 3. Serialization
            # We send the tensor + original size so the decoder can crop padding
            package = {
                "latent": latent_q.cpu(), # Send to CPU for serialization
                "orig_h": orig_h,
                "orig_w": orig_w
            }
            data = pickle.dumps(package)
            print(sys.getsizeof(data))
            """
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cv2.destroyAllWindows()

cap.release()