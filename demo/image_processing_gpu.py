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
    return tensor.float().div_(255.0).contiguous()

def postprocess_gpu(tensor: torch.Tensor) -> np.ndarray:
    """
    Optimized for MPS (Apple Silicon): RGB float32 (1, C, H, W) -> BGR uint8 (H, W, C)
    """
    with torch.no_grad():
        # 1. In-place operations reduce memory pressure on Unified Memory
        # We use clamp_ and mul_ to avoid creating intermediate tensor copies
        img = tensor.squeeze(0)  # Shape: (C, H, W)
        img = img.clamp(0, 1).mul_(255).add_(0.5)
        
        # 2. Type cast early
        img = img.to(torch.uint8)
        
        # 3. Channel Swap (RGB -> BGR)
        # On MPS, indexing or flip() are both fast, but indexing is often more explicit
        # for the compiler to optimize.
        img = img[[2, 1, 0], :, :]
        
        # 4. Permute to (H, W, C)
        img = img.permute(1, 2, 0)
        
        # 5. The "Golden Rule" for MPS -> CPU:
        # MPS tensors are often non-contiguous after a permute. 
        # Calling .contiguous() before .cpu() ensures a linear memory copy.
        return img.contiguous().cpu().numpy()

def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """RGB float32 NCHW tensor → BGR uint8 HWC numpy array."""
    img = tensor.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr