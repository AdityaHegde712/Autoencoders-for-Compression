import cv2
import torch
import pickle
import numpy as np
import sys
import time 

from confluent_kafka import Consumer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.models.autoencoder import AsymmetricAutoencoder
from scripts.live_demo import postprocess

# --- CONFIG ---
KAFKA_BROKER = 'localhost:9092'
TOPIC = 'ai-compressed-video'
CHECKPOINT = "../ml/models/saved/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model (Decoder only needed, but loading full is easier)
model = AsymmetricAutoencoder(in_channels=3, latent_channels=64).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
decoder = torch.compile(model.decoder)

c = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'ai-viewer-group',
    'auto.offset.reset': 'latest'
})
c.subscribe([TOPIC])

print("Consumer started. Decompressing latent tensors...")

try:
    with torch.no_grad():
        while True:
            msg = c.poll(0.1)
            if msg is None: continue
            print("msg receive")
            # 2. Deserialization
            package = pickle.loads(msg.value())
            latent_q = package["latent"].to(DEVICE)
            h, w = package["orig_h"], package["orig_w"]

            # 3. Decompression Pipeline
            # Decoder -> Crop Padding -> Postprocess
            x_hat = decoder(latent_q)
            x_hat = x_hat[:, :, :h, :w] # Crop to original size
            
            reconstructed_frame = postprocess(x_hat)

            cv2.imshow('AI Decompressed Feed', reconstructed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    c.close()
    cv2.destroyAllWindows()