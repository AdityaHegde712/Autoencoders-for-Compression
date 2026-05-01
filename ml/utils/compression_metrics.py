"""Compression metric helpers shared by demos and analysis scripts."""

RAW_VIDEO_BYTES_PER_PIXEL = 3
RAW_VIDEO_BITS_PER_PIXEL = RAW_VIDEO_BYTES_PER_PIXEL * 8


def raw_video_size_bytes(height: int, width: int, frames: int = 1) -> int:
    """Return uncompressed 24-bit RGB/BGR video size in bytes."""
    return height * width * frames * RAW_VIDEO_BYTES_PER_PIXEL


def compression_ratio_against_raw_video(
    compressed_bytes: float,
    height: int,
    width: int,
    frames: int = 1,
) -> float:
    """Return raw 24-bit video bytes divided by compressed bytes."""
    if compressed_bytes <= 0:
        return float("inf")
    return raw_video_size_bytes(height, width, frames) / compressed_bytes


def compression_ratio_from_bpp(bpp: float) -> float:
    """Return the compression ratio for a bits-per-pixel estimate."""
    if bpp <= 0:
        return float("inf")
    return RAW_VIDEO_BITS_PER_PIXEL / bpp
