import torch
import torch.nn as nn
from ml.models.entropy import FactorizedBottleneck

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
        # Goal: Reduce spatial dimensions (e.g., 8x)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=1) # Mapping to bottleneck
        )

        # 2. BOTTLENECK (Factorized Prior)
        self.bottleneck = FactorizedBottleneck(latent_channels)

        # 3. HEAVY DECODER (for Server side)
        # Goal: Upsample and restore detail
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(32),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
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

if __name__ == "__main__":
    # Test forward pass with dummy data
    model = AsymmetricAutoencoder(in_channels=3)
    dummy_input = torch.randn(1, 3, 256, 256)
    x_hat, p_y, y = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Latent shape: {y.shape}")
    print(f"Likelihood shape: {p_y.shape}")
