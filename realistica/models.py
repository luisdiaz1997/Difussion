"""
Diffusion model architectures
Implementation based on Denoising Diffusion Probabilistic Models (DDPM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoiseScheduler:
    """
    Noise scheduler for DDPM that manages the forward diffusion process.
    Implements both linear and cosine beta schedules.
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
        device="cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Create beta schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Pre-compute useful quantities for diffusion process
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x_0: Original images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-generated noise

        Returns:
            x_t: Noisy images at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t, noise

    def sample_timesteps(self, batch_size):
        """
        Sample random timesteps for training
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

    def add_gp_noise(self, x_0, t, gp_noise_sampler=None, lengthscale=0.1, **kwargs):
        """
        Forward diffusion process with GP-structured noise: q(x_t | x_0)

        Args:
            x_0: Original images [B, C, H, W]
            t: Timesteps [B]
            gp_noise_sampler: Optional pre-initialized GP noise sampler
            lengthscale: GP lengthscale (if sampler not provided)
            **kwargs: Additional arguments for GP sampler

        Returns:
            x_t: Noisy images at timestep t
            noise: The GP noise that was added
        """
        from .fourier_samplers import ImageGPNoiseSampler

        B, C, H, W = x_0.shape

        # Create or use provided GP sampler
        if gp_noise_sampler is None:
            gp_noise_sampler = ImageGPNoiseSampler(
                height=H,
                width=W,
                lengthscale=lengthscale,
                variance=1.0,
                num_features=kwargs.get('num_features', 1024),
                kernel_type=kwargs.get('kernel_type', 'matern32'),
                device=self.device
            )

        # Sample GP noise
        noise = gp_noise_sampler.sample(B, C)

        # Apply diffusion schedule
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t, noise


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding similar to Transformer positional encoding
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: Timesteps [B]
        Returns:
            Time embeddings [B, dim]
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x: Input tensor [B, C, H, W]
            t_emb: Time embedding [B, time_emb_dim]
        """
        h = self.conv1(x)

        # Add time conditioning
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.conv2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for U-Net
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)

        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(C // self.num_heads)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        h = torch.matmul(attn, v)

        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        h = self.proj(h)

        return x + h


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2,
                 downsample=True, attention=False):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])

        self.attention = nn.ModuleList([
            AttentionBlock(out_channels) if attention else nn.Identity()
            for _ in range(num_layers)
        ])

        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x, t_emb):
        for res_block, attn in zip(self.res_blocks, self.attention):
            x = res_block(x, t_emb)
            x = attn(x)

        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2,
                 upsample=True, attention=False):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])

        self.attention = nn.ModuleList([
            AttentionBlock(out_channels) if attention else nn.Identity()
            for _ in range(num_layers)
        ])

        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()

    def forward(self, x, t_emb):
        for res_block, attn in zip(self.res_blocks, self.attention):
            x = res_block(x, t_emb)
            x = attn(x)

        x = self.upsample(x)
        return x


class UNet(nn.Module):
    """
    U-Net denoiser model for DDPM
    Predicts the noise added to an image at timestep t
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_multipliers=(1, 2, 3, 4),
        num_res_blocks=2,
        time_emb_dim=512,
        attention_levels=(False, False, True, True),
        dropout=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embedding = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    time_emb_dim,
                    num_layers=num_res_blocks,
                    downsample=(i < len(channel_multipliers) - 1),
                    attention=attention_levels[i]
                )
            )
            in_ch = out_ch
            channels.append(in_ch)

        # Middle block
        self.middle_block = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, time_emb_dim, dropout),
            AttentionBlock(in_ch),
            ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)
        ])

        # Upsampling path
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult

            self.up_blocks.append(
                UpBlock(
                    in_ch + channels.pop(),  # Skip connection
                    out_ch,
                    time_emb_dim,
                    num_layers=num_res_blocks + 1,
                    upsample=(i < len(channel_multipliers) - 1),
                    attention=attention_levels[len(channel_multipliers) - 1 - i]
                )
            )
            in_ch = out_ch

        # Final output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        """
        Args:
            x: Noisy images [B, C, H, W]
            t: Timesteps [B]

        Returns:
            Predicted noise [B, C, H, W]
        """
        # Store input size for final output
        input_size = x.shape[-2:]

        # Time embedding
        t_emb = self.time_embedding(t)

        # Initial convolution
        x = self.conv_in(x)

        # Store skip connections
        skips = [x]

        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x, t_emb)
            skips.append(x)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)

        # Upsampling
        for up_block in self.up_blocks:
            skip = skips.pop()
            # Resize x to match skip dimensions (handles odd-sized images)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, t_emb)

        # Final output
        x = self.conv_out(x)

        # Ensure output matches input size
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='nearest')

        return x
