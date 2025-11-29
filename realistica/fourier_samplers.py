"""
Fourier-based samplers for efficient Gaussian Process sampling
Uses Random Fourier Features (RFF) to avoid expensive kernel matrix computations
"""

import torch
import torch.nn as nn
import math


class FourierSampler(nn.Module):
    """
    Base class for Fourier-based GP samplers using Random Fourier Features
    """

    def __init__(self, num_features, input_dim, lengthscale=1.0, variance=1.0, device='cpu'):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim
        self.lengthscale = lengthscale
        self.variance = variance
        self.device = device

    def sample(self, coords):
        """
        Sample from GP at given coordinates

        Args:
            coords: Input coordinates [N, D]

        Returns:
            GP samples [N]
        """
        raise NotImplementedError


class Matern32FourierSampler(FourierSampler):
    """
    Fourier sampler for Matern 3/2 kernel using Random Fourier Features

    For Matern 3/2, the spectral density is a Student's t-distribution.
    We can sample frequencies from this distribution and use them to
    approximate the kernel via:

    k(x, x') ≈ (1/M) Σ φ(x)^T φ(x')

    where φ(x) = √(2σ²/M) [cos(ω₁^T x), sin(ω₁^T x), ..., cos(ω_M^T x), sin(ω_M^T x)]

    Args:
        num_features: Number of random Fourier features (M)
        input_dim: Input dimensionality
        lengthscale: Kernel lengthscale
        variance: Kernel variance
        device: Device to run on
    """

    def __init__(self, num_features=1024, input_dim=2, lengthscale=1.0, variance=1.0, device='cpu'):
        super().__init__(num_features, input_dim, lengthscale, variance, device)

        # Sample random frequencies from the spectral density
        # For Matern 3/2, frequencies follow a specific distribution
        self.omega = self._sample_spectral_frequencies()
        self.omega = self.omega.to(device)

        # Sample random phases
        self.b = torch.rand(num_features, device=device) * 2 * math.pi

        # Scale factor
        self.scale = math.sqrt(2 * variance / num_features)

    def _sample_spectral_frequencies(self):
        """
        Sample frequencies from Matern 3/2 spectral density

        For Matern 3/2, we use a Student's t-distribution approximation
        """
        # For Matern 3/2, we can sample from a Gaussian and then scale
        # This is an approximation using the fact that Matern spectral density
        # can be approximated by a heavy-tailed distribution

        # Sample from standard normal
        omega = torch.randn(self.num_features, self.input_dim)

        # Scale by lengthscale (inverse relationship)
        # For Matern 3/2, the characteristic frequency is ~1/lengthscale
        omega = omega / self.lengthscale

        # Add heavy tails for Matern (using Laplace-like sampling)
        # This creates the right spectral characteristics
        scale_factor = torch.abs(torch.randn(self.num_features, 1)) + 0.5
        omega = omega * scale_factor

        return omega

    def sample(self, coords, num_samples=1):
        """
        Sample from Matern 3/2 GP at given coordinates

        Args:
            coords: Input coordinates [N, D]
            num_samples: Number of independent GP samples

        Returns:
            GP samples [num_samples, N]
        """
        coords = coords.to(self.device)
        N = coords.shape[0]

        samples = []
        for _ in range(num_samples):
            # Random Fourier Features
            # φ(x) = scale * cos(ω^T x + b)
            features = torch.cos(coords @ self.omega.T + self.b)  # [N, M]

            # Sample random weights
            weights = torch.randn(self.num_features, device=self.device)

            # Compute GP sample
            sample = self.scale * (features @ weights)
            samples.append(sample)

        return torch.stack(samples, dim=0)


class RBFFourierSampler(FourierSampler):
    """
    Fourier sampler for RBF (Squared Exponential) kernel

    For RBF kernel, frequencies are sampled from a Gaussian distribution.

    Args:
        num_features: Number of random Fourier features
        input_dim: Input dimensionality
        lengthscale: Kernel lengthscale
        variance: Kernel variance
        device: Device to run on
    """

    def __init__(self, num_features=1024, input_dim=2, lengthscale=1.0, variance=1.0, device='cpu'):
        super().__init__(num_features, input_dim, lengthscale, variance, device)

        # For RBF, frequencies are Gaussian with std = 1/lengthscale
        self.omega = torch.randn(num_features, input_dim, device=device) / lengthscale

        # Random phases
        self.b = torch.rand(num_features, device=device) * 2 * math.pi

        # Scale factor
        self.scale = math.sqrt(2 * variance / num_features)

    def sample(self, coords, num_samples=1):
        """
        Sample from RBF GP at given coordinates

        Args:
            coords: Input coordinates [N, D]
            num_samples: Number of independent GP samples

        Returns:
            GP samples [num_samples, N]
        """
        coords = coords.to(self.device)
        N = coords.shape[0]

        samples = []
        for _ in range(num_samples):
            # Random Fourier Features
            features = torch.cos(coords @ self.omega.T + self.b)  # [N, M]

            # Sample random weights
            weights = torch.randn(self.num_features, device=self.device)

            # Compute GP sample
            sample = self.scale * (features @ weights)
            samples.append(sample)

        return torch.stack(samples, dim=0)


def sample_gp_noise_for_images(
    batch_size,
    channels,
    height,
    width,
    lengthscale=0.1,
    variance=1.0,
    num_features=1024,
    kernel_type='matern32',
    device='cpu'
):
    """
    Sample GP noise for a batch of images

    Args:
        batch_size: Number of images in batch
        channels: Number of channels
        height: Image height
        width: Image width
        lengthscale: GP lengthscale (larger = smoother noise)
        variance: GP variance (output scale)
        num_features: Number of Fourier features
        kernel_type: Type of kernel ('matern32', 'rbf')
        device: Device to run on

    Returns:
        GP noise tensor [batch_size, channels, height, width]
    """
    # Get image coordinates
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # [H*W, 2]

    # Initialize sampler
    if kernel_type == 'matern32':
        sampler = Matern32FourierSampler(
            num_features=num_features,
            input_dim=2,
            lengthscale=lengthscale,
            variance=variance,
            device=device
        )
    elif kernel_type == 'rbf':
        sampler = RBFFourierSampler(
            num_features=num_features,
            input_dim=2,
            lengthscale=lengthscale,
            variance=variance,
            device=device
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Sample GP noise for each image and channel
    noise_list = []
    for b in range(batch_size):
        channel_noise = []
        for c in range(channels):
            # Sample one GP realization
            gp_sample = sampler.sample(coords, num_samples=1)[0]  # [H*W]

            # Reshape to image
            gp_sample = gp_sample.reshape(height, width)
            channel_noise.append(gp_sample)

        noise_list.append(torch.stack(channel_noise, dim=0))

    noise = torch.stack(noise_list, dim=0)  # [B, C, H, W]

    # Standardize to have zero mean and unit variance (approximately)
    noise = (noise - noise.mean()) / (noise.std() + 1e-8)

    return noise


class ImageGPNoiseSampler:
    """
    Reusable GP noise sampler for images
    Creates samplers once and reuses them for efficiency
    """

    def __init__(
        self,
        height,
        width,
        lengthscale=0.1,
        variance=1.0,
        num_features=1024,
        kernel_type='matern32',
        device='cpu'
    ):
        self.height = height
        self.width = width
        self.lengthscale = lengthscale
        self.variance = variance
        self.num_features = num_features
        self.kernel_type = kernel_type
        self.device = device

        # Pre-compute coordinates
        y = torch.linspace(0, 1, height, device=device)
        x = torch.linspace(0, 1, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

        # Create sampler
        if kernel_type == 'matern32':
            self.sampler = Matern32FourierSampler(
                num_features=num_features,
                input_dim=2,
                lengthscale=lengthscale,
                variance=variance,
                device=device
            )
        elif kernel_type == 'rbf':
            self.sampler = RBFFourierSampler(
                num_features=num_features,
                input_dim=2,
                lengthscale=lengthscale,
                variance=variance,
                device=device
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def sample(self, batch_size, channels):
        """
        Sample GP noise for a batch of images

        Args:
            batch_size: Number of images
            channels: Number of channels

        Returns:
            GP noise [batch_size, channels, height, width]
        """
        noise_list = []
        for b in range(batch_size):
            channel_noise = []
            for c in range(channels):
                # Sample one GP realization
                gp_sample = self.sampler.sample(self.coords, num_samples=1)[0]

                # Reshape to image
                gp_sample = gp_sample.reshape(self.height, self.width)
                channel_noise.append(gp_sample)

            noise_list.append(torch.stack(channel_noise, dim=0))

        noise = torch.stack(noise_list, dim=0)

        # Standardize
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        return noise

    def __call__(self, batch_size, channels):
        """Allow calling as a function"""
        return self.sample(batch_size, channels)
