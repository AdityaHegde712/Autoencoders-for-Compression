"""
Range coding using constriction library for learned compression.

Industry standard entropy coding used in CompressAI, JPEG XL, and learned compression.
"""

import numpy as np
import torch
import constriction
from typing import Tuple, Optional


def build_cdf_tables_from_bottleneck(bottleneck, symbol_range: int = 256) -> Tuple[np.ndarray, int]:
    """
    Build per-channel CDF tables from a FactorizedBottleneck.
    
    The bottleneck uses per-channel Laplace distributions (zero-mean, learned scale).
    This function converts the continuous distribution to discrete PMF/CDF tables
    suitable for range coding.
    
    Args:
        bottleneck: FactorizedBottleneck instance with log_scale parameter
        symbol_range: Number of possible symbol values (covers [-R/2, R/2))
        
    Returns:
        cdf_tables: Array of shape (C, symbol_range) with cumulative probabilities
        symbol_offset: Value to add to symbols before encoding (maps to [0, R))
    """
    assert symbol_range > 0
    device = bottleneck.log_scale.device
    C = bottleneck.log_scale.shape[1]  # Number of channels
    
    # Symbol range: [-half, half) where half = symbol_range // 2
    half = symbol_range // 2
    symbols = torch.arange(-half, half, dtype=torch.float32, device=device)  # Shape: (R,)
    
    # Scale shape: (1, C, 1, 1) -> (C,)
    scales = torch.exp(bottleneck.log_scale).squeeze()  # Shape: (C,)
    
    # Compute Laplace CDF for each (channel, symbol) pair
    # Laplace CDF: 0.5 + 0.5 * sign(x) * (1 - exp(-|x|/b))
    # We need P(symbol i) = CDF(i+0.5) - CDF(i-0.5)
    
    # Expand symbols for vectorized computation: (C, R)
    sym_plus_half = symbols + 0.5  # (R,) -> broadcast to (C, R)
    sym_minus_half = symbols - 0.5
    
    # Compute CDF values at each boundary
    def laplace_cdf(x, b):
        """Laplace CDF: 0.5 + 0.5 * sign(x) * (1 - exp(-|x|/b))"""
        return 0.5 + 0.5 * torch.sign(x) * (1.0 - torch.exp(-torch.abs(x) / b))
    
    # Compute PMF for each channel-symbol pair
    # Shape: (C, R) where cdf_plus and cdf_minus are (C, R)
    cdf_plus = laplace_cdf(sym_plus_half.unsqueeze(0).expand(C, -1), 
                           scales.unsqueeze(1))
    cdf_minus = laplace_cdf(sym_minus_half.unsqueeze(0).expand(C, -1), 
                              scales.unsqueeze(1))
    
    pmf = cdf_plus - cdf_minus  # Shape: (C, R)
    pmf = torch.clamp(pmf, min=1e-10)
    
    # Normalize per channel to ensure sum = 1
    pmf = pmf / pmf.sum(dim=1, keepdim=True)
    
    # Convert to CDF (cumulative sum along symbol dimension)
    cdf = torch.cumsum(pmf, dim=1)  # Shape: (C, R)
    cdf = torch.clamp(cdf, max=1.0)
    
    return cdf.detach().cpu().numpy(), half


def quantize_latents_to_symbols(y_q: torch.Tensor, symbol_offset: int, symbol_range: int) -> np.ndarray:
    """
    Convert float latents to integer symbols in [0, symbol_range).
    
    Args:
        y_q: Quantized latents (float32, shape: (B, C, H, W))
        symbol_offset: Offset to add (maps [-half, half) -> [0, R))
        symbol_range: Total number of symbols
        
    Returns:
        symbols: int32 array, flattened as (B*C*H*W,)
    """
    # Round to nearest integer (already done by round_ste, but ensure)
    symbols = torch.round(y_q).to(torch.int32)
    
    # Add offset to map to [0, symbol_range)
    symbols = symbols + symbol_offset
    
    # Clamp to valid range
    symbols = torch.clamp(symbols, 0, symbol_range - 1)
    
    return symbols.cpu().numpy().flatten()


class RangeEncoder:
    """
    Encodes quantized latent values using range coding with constriction.
    
    Uses per-channel CDF tables for proper entropy coding.
    """
    
    def __init__(self):
        self.encoder = constriction.stream.queue.RangeEncoder()
        
    def encode(self, symbols: np.ndarray, cdf_tables: np.ndarray, shape: Tuple[int, ...]) -> bytes:
        """
        Encode symbols using per-channel CDF tables (batched by channel for speed).
        
        Args:
            symbols: Flattened array of quantized latent values (int), shape (N,)
            cdf_tables: Per-channel CDF tables, shape (C, symbol_range)
            shape: Original tensor shape (B, C, H, W)
            
        Returns:
            Compressed byte stream
        """
        symbols = np.asarray(symbols).flatten().astype(np.int32)
        B, C, H, W = shape
        num_symbols = B * C * H * W
        assert len(symbols) == num_symbols, f"Symbol count mismatch: {len(symbols)} vs {num_symbols}"
        
        symbols_per_channel = B * H * W
        
        # Pre-compute PMF for each channel once
        pmf_cache = {}
        for c in range(C):
            cdf = cdf_tables[c]
            pmf = np.diff(np.concatenate([[0], cdf])).astype(np.float32)
            pmf = pmf / pmf.sum()
            pmf_cache[c] = pmf
        
        # Encode symbols grouped by channel
        for c in range(C):
            channel_symbols = symbols[c * symbols_per_channel : (c + 1) * symbols_per_channel]
            pmf = pmf_cache[c]
            model = constriction.stream.model.Categorical(pmf, perfect=False)
            self.encoder.encode(channel_symbols, model)
        
        return self.encoder.get_compressed().tobytes()
    
    def reset(self):
        """Reset encoder for new encoding"""
        self.encoder = constriction.stream.queue.RangeEncoder()


class RangeDecoder:
    """
    Decodes compressed bitstream using range coding with constriction.
    
    Uses per-channel CDF tables for proper entropy coding.
    """
    
    def __init__(self):
        self.decoder = None
        
    def decode(self, compressed: bytes, cdf_tables: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Decode compressed bitstream.
        
        Args:
            compressed: Compressed byte stream (bytes or uint32 array)
            cdf_tables: Per-channel CDF tables, shape (C, symbol_range)
            shape: Target shape for output (B, C, H, W)
            
        Returns:
            Reconstructed quantized latent values, flattened
        """
        B, C, H, W = shape
        num_symbols = B * C * H * W
        symbols_per_channel = B * H * W
        
        # Recreate decoder from compressed data
        if isinstance(compressed, bytes):
            compressed_arr = np.frombuffer(compressed, dtype=np.uint32)
        else:
            compressed_arr = np.asarray(compressed, dtype=np.uint32)
        self.decoder = constriction.stream.queue.RangeDecoder(compressed_arr)
        
        symbols = np.zeros(num_symbols, dtype=np.int32)
        
        # Pre-compute PMF for each channel once
        pmf_cache = {}
        for c in range(C):
            cdf = cdf_tables[c]
            pmf = np.diff(np.concatenate([[0], cdf])).astype(np.float32)
            pmf = pmf / pmf.sum()
            pmf_cache[c] = pmf
        
        # Decode symbols grouped by channel
        for c in range(C):
            pmf = pmf_cache[c]
            model = constriction.stream.model.Categorical(pmf, perfect=False)
            decoded = self.decoder.decode(model, symbols_per_channel)
            start = c * symbols_per_channel
            end = start + symbols_per_channel
            symbols[start:end] = decoded
        
        return symbols.reshape(shape)


def encode_frame(y_q: np.ndarray, cdf_tables: np.ndarray, shape: Tuple[int, ...]) -> bytes:
    """
    Convenience function to encode a single frame.
    
    Args:
        y_q: Quantized latent values (int, flattened)
        cdf_tables: Per-channel CDF tables
        shape: Original tensor shape
        
    Returns:
        Compressed bytes
    """
    encoder = RangeEncoder()
    return encoder.encode(y_q, cdf_tables, shape)


def decode_frame(compressed: bytes, cdf_tables: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Convenience function to decode a single frame.
    
    Args:
        compressed: Compressed bytes
        cdf_tables: Per-channel CDF tables
        shape: Target shape for output
        
    Returns:
        Reconstructed quantized latent values (flattened)
    """
    decoder = RangeDecoder()
    return decoder.decode(compressed, cdf_tables, shape)