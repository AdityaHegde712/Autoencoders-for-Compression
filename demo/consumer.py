import cv2
import torch
import numpy as np
import sys
import time 
import blosc
import struct
from safetensors.torch import load as safe_load

from confluent_kafka import Consumer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.models.autoencoder import AsymmetricAutoencoder
from ml.utils.device import get_device, maybe_compile
from scripts.live_demo import postprocess

# --- CONFIG ---
KAFKA_BROKER = 'localhost:9092'
TOPIC = 'ai-compressed-video'
CHECKPOINT = "../ml/models/saved/DWS_epoch_30.pth"
DEVICE = get_device()

RESOLUTION_W = 1920
RESOLUTION_H = 1080

# --- HELPER ---
def bitstream_to_tensor(bitstream, shape):
    """
    byte_stream: The compressed bytes
    log_scale: The learned parameters from your model
    shape: The spatial shape of the latent (B, C, H, W)
    """
    decompressed_bytes = blosc.decompress(bitstream)
    flat_data = np.frombuffer(decompressed_bytes, dtype=np.float32)
    tensor = torch.from_numpy(flat_data).reshape(shape)

    return tensor

def unpackage_frame(packet):
    # Header is 21 bytes (4+1+4+4+4+4)
    header_size = struct.calcsize(">I?IIII")
    header_data = packet[:header_size]
    bitstream = packet[header_size:]
    
    # Unpack header
    frame_idx, is_iframe, c, h, w, payload_len = struct.unpack(">I?IIII", header_data)
    
    return {
        "idx": frame_idx,
        "is_iframe": is_iframe,
        "shape": (1, c, h, w),
        "bitstream": bitstream
    }

# 1. Load Model (Decoder only needed, but loading full is easier)
model = AsymmetricAutoencoder(in_channels=3, latent_channels=64).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
model.eval()
decoder = maybe_compile(model.decoder, DEVICE)

c = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'ai-viewer-group',
    'auto.offset.reset': 'latest'
})
c.subscribe([TOPIC])

print("Consumer started. Decompressing latent tensors...")

try:
    with torch.inference_mode():
        frame_count = 0
        iteration_times = []
        deser_times = []
        decode_times = []
        postproc_times = []
        
        while True:
            msg = c.poll(0.1)
            if msg is None: continue
            
            # --- TIMING: Start iteration timer ---
            iteration_start = time.time()
            
            package = unpackage_frame(msg.value())
            

            # 2. Deserialization
            deser_start = time.time()
            latent_q = bitstream_to_tensor(package["bitstream"], package["shape"]).to(
                DEVICE, non_blocking=DEVICE.type == "cuda"
            )
            deser_time = time.time() - deser_start
            
            #latent_q = package["latent"].to(DEVICE)
            h, w = RESOLUTION_H, RESOLUTION_W
            
            # 3. Decompression Pipeline
            # Decoder -> Crop Padding -> Postprocess
            decode_start = time.time()
            x_hat = decoder(latent_q)
            x_hat = x_hat[:, :, :h, :w] # Crop to original size
            decode_time = time.time() - decode_start
            
            postproc_start = time.time()
            reconstructed_frame = postprocess(x_hat)
            postproc_time = time.time() - postproc_start

            cv2.imshow('AI Decompressed Feed', reconstructed_frame)
            
            # --- TIMING: Collect iteration time ---
            iteration_time = time.time() - iteration_start
            frame_count += 1
            
            iteration_times.append(iteration_time)
            deser_times.append(deser_time)
            decode_times.append(decode_time)
            postproc_times.append(postproc_time)
            
            # Print after every 10 iterations
            if frame_count % 10 == 0:
                avg_iter = (sum(iteration_times) / len(iteration_times)) * 1000
                avg_deser = (sum(deser_times) / len(deser_times)) * 1000
                avg_decode = (sum(decode_times) / len(decode_times)) * 1000
                avg_postproc = (sum(postproc_times) / len(postproc_times)) * 1000
                
                print(f"\n--- Frames {frame_count-9} to {frame_count} Avg Timing ---")
                print(f"  Deserialization: {avg_deser:.2f}ms")
                print(f"  Decoding: {avg_decode:.2f}ms")
                print(f"  Postprocessing: {avg_postproc:.2f}ms")
                print(f"  Total Iteration: {avg_iter:.2f}ms\n")
                
                # Reset accumulators
                iteration_times = []
                deser_times = []
                decode_times = []
                postproc_times = []
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    c.close()
    cv2.destroyAllWindows()