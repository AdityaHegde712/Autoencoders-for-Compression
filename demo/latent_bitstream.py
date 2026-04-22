import blosc

from ml.utils.ops import round_ste


def convert_to_bitstream(tensor):
    # 1. Quantize
    quantized = round_ste(tensor)

    # 2. Convert to bytes and compress
    raw_bytes = quantized.detach().cpu().numpy().tobytes()
    bitstream = blosc.compress(raw_bytes, cname="zstd", clevel=5)

    return bitstream
