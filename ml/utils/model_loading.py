"""
Shared model loading utilities.
Centralizes detect_latent_channels and load_model to eliminate
identical 3-way copies across evaluate.py, visualize.py, and test_code.py.
"""
import torch

from ml.utils.constants import get_device


def detect_latent_channels(model_path):
    """
    Detect latent_channels from a saved checkpoint by inspecting
    the bottleneck or encoder state dict keys.
    """
    state_dict = torch.load(model_path, map_location='cpu')

    # Preferred: bottleneck.log_scale stores per-channel scale
    if 'bottleneck.log_scale' in state_dict:
        return state_dict['bottleneck.log_scale'].shape[1]

    # Fallbacks for architectures where encoder keys differ
    keys_to_check = ['encoder.4.weight', 'encoder.7.weight', 'encoder.8.weight']
    for key in keys_to_check:
        if key in state_dict:
            return state_dict[key].shape[0]

    raise ValueError(f"Cannot detect latent_channels from checkpoint {model_path}")


def load_model(model_path, latent_channels, device=None, verbose=True):
    """
    Load the correct model architecture based on the run path.

    Routes to Gabriel (depthwise), Jasper (lightweight), or the
    default Best (AsymmetricAutoencoder) based on keyword matching
    in the path.  This avoids hard-coding architecture choices at
    the call site.
    """
    if device is None:
        device = get_device()

    model_path_lower = str(model_path).lower()
    model_type = "Original (autoencoder)"

    if "gabriel" in model_path_lower:
        from ml.models.iframe_encoder import AsymmetricAutoencoder as GabrielModel
        legacy = (latent_channels == 64)
        model = GabrielModel(
            in_channels=3, latent_channels=latent_channels,
            legacy_encoder=legacy,
        ).to(device)
        model_type = "Gabriel (iframe_encoder)"
    elif "jasper" in model_path_lower:
        from ml.models.autoencoder import AsymmetricAutoencoder2 as JasperModel
        model = JasperModel(
            in_channels=3, latent_channels=latent_channels,
        ).to(device)
        model_type = "Jasper (autoencoder AsymmetricAutoencoder2)"
    else:
        from ml.models.autoencoder import AsymmetricAutoencoder as BestModel
        model = BestModel(
            in_channels=3, latent_channels=latent_channels,
        ).to(device)
        model_type = "Original (autoencoder)"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if verbose:
        print(f"[Model] Loaded {model_type} from {model_path} on {device}")

    return model
