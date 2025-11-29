"""
Kernel functions for Gaussian Processes
"""

import torch
import math


class Kernel:
    """
    Base class for kernel functions
    """

    def __call__(self, x1, x2):
        """
        Compute kernel matrix K(x1, x2)

        Args:
            x1: Input points [N1, D]
            x2: Input points [N2, D]

        Returns:
            Kernel matrix [N1, N2]
        """
        raise NotImplementedError


class Matern32(Kernel):
    """
    Matern 3/2 kernel: k(x, x') = σ² (1 + √3 r/ℓ) exp(-√3 r/ℓ)
    where r = ||x - x'||

    This kernel produces once-differentiable sample paths and is
    commonly used for spatial modeling.

    Args:
        lengthscale: Length scale parameter (controls smoothness)
        variance: Variance parameter (output scale)
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1, x2):
        """
        Compute Matern 3/2 kernel matrix

        Args:
            x1: Input points [N1, D]
            x2: Input points [N2, D]

        Returns:
            Kernel matrix [N1, N2]
        """
        # Compute pairwise distances
        x1 = x1.unsqueeze(1)  # [N1, 1, D]
        x2 = x2.unsqueeze(0)  # [1, N2, D]
        r = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))  # [N1, N2]

        # Matern 3/2 formula
        sqrt3_r_over_l = math.sqrt(3) * r / self.lengthscale
        k = self.variance * (1 + sqrt3_r_over_l) * torch.exp(-sqrt3_r_over_l)

        return k

    def spectral_density(self, omega):
        """
        Spectral density for Random Fourier Features
        For Matern 3/2: S(ω) ∝ (1 + ||ω||²/3ℓ²)^(-2)

        Args:
            omega: Frequencies [M, D]

        Returns:
            Spectral density values [M]
        """
        omega_norm_sq = torch.sum(omega ** 2, dim=-1)
        nu = 1.5  # For Matern 3/2
        return self.variance * (2 ** (omega.shape[-1]) * math.pi ** (omega.shape[-1] / 2) *
                                math.gamma(nu + omega.shape[-1] / 2) * (2 * nu) ** nu /
                                (math.gamma(nu) * self.lengthscale ** (2 * nu)) *
                                (2 * nu / self.lengthscale ** 2 + omega_norm_sq) ** (-(nu + omega.shape[-1] / 2)))


class RBF(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential kernel
    k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))

    Args:
        lengthscale: Length scale parameter
        variance: Variance parameter
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1, x2):
        """
        Compute RBF kernel matrix

        Args:
            x1: Input points [N1, D]
            x2: Input points [N2, D]

        Returns:
            Kernel matrix [N1, N2]
        """
        # Compute pairwise squared distances
        x1 = x1.unsqueeze(1)  # [N1, 1, D]
        x2 = x2.unsqueeze(0)  # [1, N2, D]
        r_sq = torch.sum((x1 - x2) ** 2, dim=-1)  # [N1, N2]

        # RBF formula
        k = self.variance * torch.exp(-r_sq / (2 * self.lengthscale ** 2))

        return k


class Matern52(Kernel):
    """
    Matern 5/2 kernel: k(x, x') = σ² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(-√5 r/ℓ)

    This kernel produces twice-differentiable sample paths.

    Args:
        lengthscale: Length scale parameter
        variance: Variance parameter
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1, x2):
        """
        Compute Matern 5/2 kernel matrix

        Args:
            x1: Input points [N1, D]
            x2: Input points [N2, D]

        Returns:
            Kernel matrix [N1, N2]
        """
        # Compute pairwise distances
        x1 = x1.unsqueeze(1)  # [N1, 1, D]
        x2 = x2.unsqueeze(0)  # [1, N2, D]
        r = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))  # [N1, N2]

        # Matern 5/2 formula
        sqrt5_r_over_l = math.sqrt(5) * r / self.lengthscale
        k = self.variance * (1 + sqrt5_r_over_l + (5 * r ** 2) / (3 * self.lengthscale ** 2)) * \
            torch.exp(-sqrt5_r_over_l)

        return k


def get_image_coordinates(height, width, device='cpu'):
    """
    Get normalized 2D coordinates for an image

    Args:
        height: Image height
        width: Image width
        device: Device to create tensor on

    Returns:
        Coordinates tensor [height * width, 2] with values in [0, 1]
    """
    # Create meshgrid
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Flatten and stack
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

    return coords
