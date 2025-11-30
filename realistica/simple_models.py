"""
Simple baseline models for diffusion
These are much simpler than U-Nets and work well for MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleMLP(nn.Module):
    """
    Super simple MLP for MNIST diffusion
    Just flattens image, concatenates time, and predicts noise

    This is essentially what the original diffusion papers used for simple datasets
    """

    def __init__(self, image_size=28, hidden_dim=512, time_emb_dim=128):
        super().__init__()
        self.image_size = image_size
        img_channels = 1
        input_dim = img_channels * image_size * image_size

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Main MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        """
        Args:
            x: [B, 1, H, W] images
            t: [B] timesteps
        Returns:
            [B, 1, H, W] predicted noise
        """
        batch_size = x.shape[0]

        # Flatten image
        x_flat = x.view(batch_size, -1)  # [B, 784]

        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        # Concatenate
        h = torch.cat([x_flat, t_emb], dim=1)  # [B, 784 + time_emb_dim]

        # Predict noise
        noise_flat = self.mlp(h)  # [B, 784]

        # Reshape back
        noise = noise_flat.view(batch_size, 1, self.image_size, self.image_size)

        return noise


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST diffusion - better than MLP but simpler than U-Net
    No skip connections, just a basic encoder-decoder
    """

    def __init__(self, image_size=28, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.image_size = image_size

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
        )

        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )

        # Middle - time conditioning
        self.mid_time = nn.Linear(time_emb_dim, base_channels * 2)

        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )

        # Decoder (upsampling)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, x, t):
        """
        Args:
            x: [B, 1, H, W] images
            t: [B] timesteps
        Returns:
            [B, 1, H, W] predicted noise
        """
        # Store input size
        input_size = x.shape[-2:]

        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        # Encode
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)

        # Add time conditioning
        t_cond = self.mid_time(t_emb)[:, :, None, None]
        h = h3 + t_cond

        # Middle
        h = self.mid(h)

        # Decode
        h = self.dec3(h)
        h = self.dec2(h)
        h = self.dec1(h)

        # Ensure output size matches input
        if h.shape[-2:] != input_size:
            h = nn.functional.interpolate(h, size=input_size, mode='nearest')

        return h


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for timesteps
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleConvDenoiser(nn.Module):
    """
    Super simple convolutional autoencoder for MNIST
    Just like a VAE encoder-decoder but without the VAE parts

    NO time embedding - just learns to denoise
    Perfect for testing if diffusion works at all
    """

    def __init__(self, image_size=28, latent_dim=64):
        super().__init__()

        # Encoder: 28x28 -> 14x14 -> 7x7 -> latent
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # 14x14 -> 7x7
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 7x7 -> 4x4
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Decoder: latent -> 7x7 -> 14x14 -> 28x28
        self.decoder = nn.Sequential(
            # 4x4 -> 7x7
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x, t=None):
        """
        Args:
            x: Noisy images [B, 1, 28, 28]
            t: Timesteps (IGNORED - we don't use it!)
        Returns:
            Predicted noise [B, 1, 28, 28]
        """
        # Encode
        h = self.encoder(x)

        # Decode
        out = self.decoder(h)

        # Make sure output is same size as input
        if out.shape != x.shape:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return out


class TinyConvDenoiser(nn.Module):
    """
    Even tinier - just 2 conv layers down, 2 up
    Absolute minimal model for testing
    """

    def __init__(self):
        super().__init__()

        # Down
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
        )

        # Up
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x, t=None):
        """No time conditioning - same network for all timesteps"""
        h = self.down1(x)
        h = self.down2(h)
        h = self.up2(h)
        h = self.up1(h)
        return h


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
