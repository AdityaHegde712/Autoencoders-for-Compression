import torch
import torch.nn as nn
import os
from ml.models.entropy import FactorizedBottleneck


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise + pointwise factorization of a standard Conv2d.
    Reduces MACs from (K² · Cin · Cout · H · W) to (K² · Cin · H · W) + (Cin · Cout · H · W).
    Used in the encoder to minimise compute on the edge device.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResConvBlock2(nn.Module):
    """ Residual Convolutional Block for the Heavy Decoder """
    def __init__(self, channels):
        super(ResConvBlock2, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

class AsymmetricAutoencoder2(nn.Module):
    def __init__(self, in_channels=3, latent_channels=128):
        super(AsymmetricAutoencoder2, self).__init__()
        
        # 1. SHALLOW ENCODER (for Edge device)
        # Goal: Reduce spatial dimensions (e.g., 8x)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, kernel_size=1) # Mapping to bottleneck
        )

        # 2. BOTTLENECK (Factorized Prior)
        self.bottleneck = FactorizedBottleneck(latent_channels)

        # 3. HEAVY DECODER (for Server side)
        # Goal: Upsample and restore detail
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock2(32),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock2(16),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, training=True):
        """
        Input x: Residual Frame Ri = I_t - I_{t-1}
        Returns:
            x_hat: Reconstructed Residual
            p_y: Likelihood for rate estimation
        """
        # Encoder
        y = self.encoder(x)
        
        # Bottleneck (includes quantization or noise simulation)
        y_q, p_y = self.bottleneck(y, training=training)
        
        # Decoder
        x_hat = self.decoder(y_q)
        
        return x_hat, p_y, y


class ResConvBlock(nn.Module):
    """ Residual Convolutional Block for the Heavy Decoder """
    def __init__(self, channels):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual


class AsymmetricAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=128):
        super(AsymmetricAutoencoder, self).__init__()
        
        # 1. SHALLOW ENCODER (for Edge device)
        # Goal: Reduce spatial dimensions (e.g., 4x or 8x)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=1) # Mapping to bottleneck
        )

        # 2. BOTTLENECK (Factorized Prior)
        self.bottleneck = FactorizedBottleneck(latent_channels)

        # 3. HEAVY DECODER (for Server side)
        # Goal: Upsample and restore detail
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(64),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def load_weight(self, path):
        """ Loads weights from a .pth file """
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)
        print(f"Weights loaded successfully from {path}")

    def forward(self, x, training=True):
        """
        Input x: Residual Frame Ri = I_t - I_{t-1}
        Returns:
            x_hat: Reconstructed Residual
            p_y: Likelihood for rate estimation
        """
        # Encoder
        y = self.encoder(x)
        
        # Bottleneck (includes quantization or noise simulation)
        y_q, p_y = self.bottleneck(y, training=training)
        
        # Decoder
        x_hat = self.decoder(y_q)
        
        return x_hat, p_y, y


if __name__ == "__main__":
    # Test forward pass with dummy data
    model = AsymmetricAutoencoder(in_channels=3)
    dummy_input = torch.randn(1, 3, 256, 256)
    x_hat, p_y, y = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Latent shape: {y.shape}")
    print(f"Likelihood shape: {p_y.shape}")

    # Test weight loading
    weight_path = "./saved/best/best_model.pth"
    if os.path.exists(weight_path):
        model.load_weight(weight_path)
    else:
        print(f"Weight file not found at {weight_path}")
