import torch
import torch.nn as nn
import numpy as np
from ml.utils.ops import quantize_with_noise, round_ste

class FactorizedBottleneck(nn.Module):
    """
    Learned Probability Model (Per-Channel Factorized Prior).
    Predicts the distribution of each latent element y.
    Used for Rate-Distortion Loss (at training) and Arithmetic Coding (at inference).
    """
    def __init__(self, channels):
        super(FactorizedBottleneck, self).__init__()
        # Simplified: Use a learnable Log-Standard-Deviation per channel
        # We assume the distribution is a Zero-mean Gaussian/Laplace
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def likelihood(self, y_quantized):
        """
        Computes the likelihood P(y_quantized) for rate estimation.
        L = -log2(p(y))
        """
        # Assume a standard Laplace: p(y) = 1/(2b) * exp(-|y|/b) where b = exp(log_scale)
        scale = torch.exp(self.log_scale)
        # Using a continuous approximation:
        # P(y) = Integrate_{-0.5}^{0.5} p(y+x) dx approx p(y) for small delta
        # But for training robustness, we use the CDF:
        # P(y \in [y-0.5, y+0.5]) = CDF(y+0.5) - CDF(y-0.5)
        
        y_plus = y_quantized + 0.5
        y_minus = y_quantized - 0.5
        
        # Laplace CDF: 0.5 + 0.5 * sign(x) * (1 - exp(-|x|/b))
        def laplace_cdf(x, b):
            return 0.5 + 0.5 * torch.sign(x) * (1.0 - torch.exp(-torch.abs(x) / b))

        p_y = laplace_cdf(y_plus, scale) - laplace_cdf(y_minus, scale)
        
        # Avoid zero probabilities
        p_y = torch.clamp(p_y, min=1e-9)
        return p_y

    def forward(self, y, training=True):
        if training:
            # Simulate quantization with noise
            y_quantized = quantize_with_noise(y)
        else:
            # Real quantization with STE
            y_quantized = round_ste(y)
            
        p_y = self.likelihood(y_quantized)
        return y_quantized, p_y
