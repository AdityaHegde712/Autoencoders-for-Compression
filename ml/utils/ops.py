import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def round_ste(x):
    """
    Straight-Through Estimator (STE) for rounding.
    It rounds in the forward pass but acts as an identity in the backward pass.
    """
    return (torch.round(x) - x).detach() + x

def quantize_with_noise(x):
    """
    Adds uniform noise during training to simulate quantization.
    Recommended for training entropy models (Ballé et al.)
    """
    noise = torch.rand_like(x) - 0.5
    return x + noise

def get_bpp_estimate(likelihoods, num_pixels):
    """
    Estimates bits-per-pixel (BPP) from likelihoods.
    L = -log2(p(y))
    BPP = sum(L) / num_pixels
    """
    bits = -torch.log2(likelihoods + 1e-9)
    return torch.sum(bits) / num_pixels
