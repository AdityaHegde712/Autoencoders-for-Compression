import torch
import torch.nn as nn
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

class DepthwiseSeparableConvTranspose(nn.Module):
    """
    Depthwise + pointwise factorization of a standard Conv2d.
    Reduces MACs from (K² · Cin · Cout · H · W) to (K² · Cin · H · W) + (Cin · Cout · H · W).
    Used in the encoder to minimise compute on the edge device.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False, output_padding=output_padding
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


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
    def __init__(self, in_channels=3, latent_channels=64, legacy_encoder=False):
        super(AsymmetricAutoencoder, self).__init__()
        
        # 1. SHALLOW ENCODER (for Edge device)
        if legacy_encoder:
            # Matches older checkpoints (encoder.0 / encoder.2 are plain Conv2d).
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, latent_channels, kernel_size=1),
            )
        else:
            # Depthwise separable convolutions to minimise MACs on edge.
            self.encoder = nn.Sequential(
                #block 1
                DepthwiseSeparableConv(in_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                #block 2
                DepthwiseSeparableConv(16, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                #block 3
                DepthwiseSeparableConv(32, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                #block 4
                nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, groups=64, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        # 2. BOTTLENECK (Factorized Prior)
        self.bottleneck = FactorizedBottleneck(64)

        # 3. HEAVY DECODER (for Server side)
        # Goal: Upsample and restore detail
        if legacy_encoder:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_channels, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                ResConvBlock(32),
                nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                ResConvBlock(16),
                nn.Conv2d(16, in_channels, kernel_size=3, padding=1)
            )
        else:
            self.decoder = nn.Sequential(
                #block 1
                nn.Conv2d(64, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                #block 2
                DepthwiseSeparableConvTranspose(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                #Block 3
                DepthwiseSeparableConvTranspose(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                ResConvBlock(32),
                #block 4
                DepthwiseSeparableConvTranspose(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                ResConvBlock(16),
                #block 5 — fourth ×2 to mirror encoder (four stride-2 downs); without this, output is H/2 × W/2
                DepthwiseSeparableConvTranspose(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                #block 6
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

if __name__ == "__main__":
    # Test forward pass with dummy data
    model = AsymmetricAutoencoder(in_channels=3)
    dummy_input = torch.randn(1, 3, 256, 256)
    x_hat, p_y, y = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Latent shape: {y.shape}")
    print(f"Likelihood shape: {p_y.shape}")
