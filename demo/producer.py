import cv2
import torch
import pickle
import sys
import time 

from confluent_kafka import Producer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.models.autoencoder import AsymmetricAutoencoder 
from scripts.live_demo import preprocess, pad_to_multiple

# --- CONFIG ---
KAFKA_BROKER = 'localhost:9092'
TOPIC = 'ai-compressed-video'
CHECKPOINT = "../ml/models/saved/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model
model = AsymmetricAutoencoder(in_channels=3, latent_channels=64).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
encoder = torch.compile(model.encoder)

p = Producer({'bootstrap.servers': KAFKA_BROKER, 'message.max.bytes': 5000000})
cap = cv2.VideoCapture(0)

print("Producer started. Sending AI-compressed latent tensors...")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. Compression Pipeline
        # Preprocess -> Pad -> Encode -> Bottleneck
        frame = cv2.resize(frame, (640, 480))
        tensor = preprocess(frame, DEVICE)
        tensor_padded, orig_h, orig_w = pad_to_multiple(tensor, multiple=4)
        
        latent = encoder(tensor_padded)
        latent_q, _ = model.bottleneck(latent, training=False)
        real_size = latent_q.element_size() * latent_q.nelement()
        print(latent_q.size())
        print(real_size)


        # 3. Serialization
        # We send the tensor + original size so the decoder can crop padding
        package = {
            "latent": latent_q.cpu(), # Send to CPU for serialization
            "orig_h": orig_h,
            "orig_w": orig_w
        }
        data = pickle.dumps(package)
        print(sys.getsizeof(data))



        #p.produce(TOPIC, data, headers=[("shape", str(latent_q.shape))])
        #p.poll(0) # Serve delivery callbacks

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()