import torch
import numpy as np


def preprocess_gpu(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    # 1. Move raw uint8 data to GPU immediately (fastest transfer)
    # Use non_blocking=True if your input is in pinned memory
    tensor = torch.from_numpy(frame_bgr).to(device)

    # 2. Reshape: HWC -> CHW and add batch dimension (1, C, H, W)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    # 3. BGR to RGB: flip channel dimension (C)
    # This is faster on GPU than cv2.cvtColor on CPU.
    tensor = tensor.flip(1)

    # 4. Convert to float and normalize [0, 1] on the GPU
    return tensor.float().div_(255.0)
