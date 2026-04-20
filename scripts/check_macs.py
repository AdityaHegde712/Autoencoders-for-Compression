"""
check_macs.py
Profiles the encoder's MAC operations and parameter count.
Run this before and after the depthwise separable conv change to confirm reduction.

Usage:
    python scripts/check_macs.py               # profiles full model
    python scripts/check_macs.py --encoder-only # profiles encoder submodule only
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from thop import profile, clever_format
from ml.models.autoencoder import AsymmetricAutoencoder

LATENT_CHANNELS = 32          # Must match train.py
INPUT_SHAPE = (1, 3, 352, 352)  # Must match training resolution


def main():
    parser = argparse.ArgumentParser(description="Profile MACs and parameter count.")
    parser.add_argument("--encoder-only", action="store_true",
                        help="Profile only the encoder submodule instead of the full model.")
    args = parser.parse_args()

    model = AsymmetricAutoencoder(in_channels=3, latent_channels=LATENT_CHANNELS)
    model.eval()

    dummy = torch.randn(*INPUT_SHAPE)

    if args.encoder_only:
        target = model.encoder
        macs, params = profile(target, inputs=(dummy,), verbose=False)
        label = "Encoder Only"
    else:
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        label = "Full Model"

    macs_str, params_str = clever_format([macs, params], "%.3f")

    print(f"\n{'=' * 40}")
    print(f"  {label}")
    print(f"  MACs:   {macs_str}")
    print(f"  Params: {params_str}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
