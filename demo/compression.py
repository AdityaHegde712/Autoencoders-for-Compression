import blosc
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils.ops import round_ste

def convert_to_bitstream(tensor):
    # 1. Quantize 
    quantized = round_ste(tensor)

    # 2. Convert to bytes and compress
    raw_bytes = quantized.numpy().tobytes()
    bitstream = blosc.compress(raw_bytes, cname='zstd', clevel=5)

    return bitstream