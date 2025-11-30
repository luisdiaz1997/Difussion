"""
Realistica - A diffusion model package for realistic image generation
"""

__version__ = "0.1.0"

# Import main components
from .models import (
    NoiseScheduler,
    UNet,
    TimeEmbedding,
    ResidualBlock,
    AttentionBlock,
    DownBlock,
    UpBlock
)

# Simple models for experimentation
from .simple_models import (
    SimpleMLP,
    SimpleCNN,
    SimpleConvDenoiser,
    TinyConvDenoiser,
    SinusoidalPosEmb
)

# Training utilities
from .training_utils import (
    DDPMTrainer,
    sample_ddpm,
    sample_ddpm_with_trajectory,
    visualize_diffusion_forward,
    EMA,
    count_parameters
)

# Kernels for Gaussian Processes
from .kernels import (
    Matern32,
    Matern52,
    RBF,
    get_image_coordinates
)

# Fourier samplers for GP noise
from .fourier_samplers import (
    Matern32FourierSampler,
    RBFFourierSampler,
    ImageGPNoiseSampler,
    sample_gp_noise_for_images
)

# Data utilities (to be implemented)
# from .data_utils import *

__all__ = [
    # Models
    'NoiseScheduler',
    'UNet',
    'TimeEmbedding',
    'ResidualBlock',
    'AttentionBlock',
    'DownBlock',
    'UpBlock',
    # Simple Models
    'SimpleMLP',
    'SimpleCNN',
    'SimpleConvDenoiser',
    'TinyConvDenoiser',
    'SinusoidalPosEmb',
    # Training
    'DDPMTrainer',
    'sample_ddpm',
    'sample_ddpm_with_trajectory',
    'visualize_diffusion_forward',
    'EMA',
    'count_parameters',
    # Kernels
    'Matern32',
    'Matern52',
    'RBF',
    'get_image_coordinates',
    # Fourier Samplers
    'Matern32FourierSampler',
    'RBFFourierSampler',
    'ImageGPNoiseSampler',
    'sample_gp_noise_for_images',
]
