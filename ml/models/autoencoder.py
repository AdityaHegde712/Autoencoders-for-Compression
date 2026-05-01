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
        # Uses depthwise separable convolutions to minimise MACs
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, kernel_size=1)  # 1x1 is already pointwise
        )

        # 2. BOTTLENECK (Factorized Prior)
        self.bottleneck = FactorizedBottleneck(latent_channels)

        # 3. HEAVY DECODER (for Server side)
        # Goal: Upsample and restore detail
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(32),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(16),
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


class LegacyAsymmetricAutoencoder(nn.Module):
    """Autoencoder variant used by older checkpoints with regular encoder convs."""
    def __init__(self, in_channels=3, latent_channels=128, encoder_channels=(16, 32), decoder_channels=(32, 16)):
        super(LegacyAsymmetricAutoencoder, self).__init__()
        enc_1, enc_2 = encoder_channels
        dec_1, dec_2 = decoder_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, enc_1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_1, enc_2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_2, latent_channels, kernel_size=1)
        )

        self.bottleneck = FactorizedBottleneck(latent_channels)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, dec_1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(dec_1),
            nn.ConvTranspose2d(dec_1, dec_2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            ResConvBlock(dec_2),
            nn.Conv2d(dec_2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, training=True):
        y = self.encoder(x)
        y_q, p_y = self.bottleneck(y, training=training)
        x_hat = self.decoder(y_q)
        return x_hat, p_y, y


class GabrielIFrameAutoencoder(nn.Module):
    """I-frame autoencoder architecture used by Gabriel's full-frame checkpoint."""
    def __init__(
        self,
        in_channels=3,
        latent_channels=96,
        base_channels=64,
        hidden_channels=128,
        encoder_res_blocks=3,
        decoder_res_blocks=3,
    ):
        super(GabrielIFrameAutoencoder, self).__init__()

        encoder_layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        ]
        encoder_layers.extend(ResConvBlock(hidden_channels) for _ in range(encoder_res_blocks))
        encoder_layers.append(nn.Conv2d(hidden_channels, latent_channels, kernel_size=5, stride=2, padding=2))
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck = FactorizedBottleneck(latent_channels)

        decoder_layers = []
        decoder_layers.extend(ResConvBlock(latent_channels) for _ in range(decoder_res_blocks))
        decoder_layers.extend([
            nn.Conv2d(latent_channels, hidden_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, training=True):
        y = self.encoder(x)
        y_q, p_y = self.bottleneck(y, training=training)
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
