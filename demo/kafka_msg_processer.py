import blosc
import torch
import numpy as np
from safetensors.torch import save
from ml.utils.ops import round_ste

COMPRESSOR = "zstd"
COMPRESS_LEVEL = 3
COMPRESS_SHUFFLE = blosc.BITSHUFFLE


def compress_for_kafka(mps_tensor):
    # 1. STE Quantization
    quantized = round_ste(mps_tensor)
    # 2. Convert to actual INT8 for the move to CPU
    cpu_tensor = quantized.to(torch.int8).to("cpu")
    
    # 3. Use direct memory view for speed
    # We use numpy to get a raw buffer without Safetensors overhead
    raw_bytes = cpu_tensor.numpy().tobytes()
    
    # 4. Multi-threaded Bitshuffled Compression
    # LZ4 is significantly faster than Zstd for real-time streaming
    compressed_payload = blosc.compress(
        raw_bytes,
        cname=COMPRESSOR,          # zstd: good size/speed tradeoff
        clevel=COMPRESS_LEVEL,     # level 3 is a common sweet spot
        shuffle=COMPRESS_SHUFFLE,
        typesize=1
    )
    return compressed_payload

def decompress_from_kafka(payload, shape, dtype=np.int8):
    """
    payload: The raw bytes from Kafka
    shape: The original shape (e.g., [1, 3, 224, 224])
    dtype: The numpy dtype used before compression
    """
    # 1. Blosc Decompression
    # Blosc stores the compression metadata in the byte header, 
    # so it knows it used Bitshuffle/Zstd automatically.
    decompressed_bytes = blosc.decompress(payload)
    
    # 2. Map bytes back to a Numerical Array
    # We use frombuffer for zero-copy speed
    flat_array = np.frombuffer(decompressed_bytes, dtype=dtype)
    
    # 3. Reshape and convert to Torch
    # .copy() is optional, but recommended if you plan to 
    # modify the tensor on the consumer side.
    reshaped_array = flat_array.reshape(shape)
    tensor = torch.from_numpy(reshaped_array.copy())
    
    # 4. Optional: Cast back to float if your model expects it
    return tensor.to(torch.float32)


def compress_frame_for_kafka(frame: np.ndarray) -> bytes:
    """
    Compress a uint8 HWC frame (typically BGR from OpenCV) with blosc+zstd.
    """
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frame, got {frame.dtype}")
    return blosc.compress(
        frame.tobytes(),
        cname=COMPRESSOR,
        clevel=COMPRESS_LEVEL,
        shuffle=COMPRESS_SHUFFLE,
        typesize=1,
    )


def decompress_frame_from_kafka(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    """
    Decompress blosc+zstd payload back to uint8 HWC frame.
    """
    raw = blosc.decompress(payload)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)