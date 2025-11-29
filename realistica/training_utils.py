"""
Training utilities and helpers for diffusion models
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
from typing import Optional, Callable, Dict, Any
from torch.utils.data import DataLoader


class DDPMTrainer:
    """
    Trainer class for DDPM models
    """

    def __init__(
        self,
        model: nn.Module,
        noise_scheduler,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        criterion: Optional[nn.Module] = None
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion or nn.MSELoss()

        self.global_step = 0
        self.epoch = 0
        self.losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch

        Args:
            dataloader: Training data loader

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Handle both (images, labels) and images only
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            batch_size = images.shape[0]

            # Sample random timesteps
            t = self.noise_scheduler.sample_timesteps(batch_size)

            # Add noise to images (forward diffusion)
            noisy_images, noise = self.noise_scheduler.add_noise(images, t)

            # Predict noise
            predicted_noise = self.model(noisy_images, t)

            # Calculate loss
            loss = self.criterion(predicted_noise, noise)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            self.losses.append(loss.item())
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = epoch_loss / len(dataloader)
        self.epoch += 1
        return avg_loss

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_interval: int = 5,
        sample_fn: Optional[Callable] = None,
        sample_interval: int = 5,
        scheduler: Optional[Any] = None
    ) -> Dict[str, list]:
        """
        Full training loop

        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints (if None, no saving)
            save_interval: Save checkpoint every N epochs
            sample_fn: Optional function to call for sampling (signature: fn(trainer) -> None)
            sample_interval: Call sample_fn every N epochs
            scheduler: Optional learning rate scheduler

        Returns:
            Dictionary with training metrics
        """
        epoch_losses = []

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {self.epoch} - Average Loss: {avg_loss:.6f}")

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"Learning rate: {current_lr:.6e}")

            # Save checkpoint
            if save_dir and (self.epoch % save_interval == 0):
                self.save_checkpoint(save_dir, f"checkpoint_epoch_{self.epoch}.pt")

            # Generate samples
            if sample_fn and (self.epoch % sample_interval == 0):
                sample_fn(self)

        # Save final checkpoint
        if save_dir:
            self.save_checkpoint(save_dir, "final_model.pt")

        return {
            'epoch_losses': epoch_losses,
            'step_losses': self.losses
        }

    def save_checkpoint(self, save_dir: str, filename: str):
        """
        Save model checkpoint

        Args:
            save_dir: Directory to save checkpoint
            filename: Checkpoint filename
        """
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, filename)

        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.losses = checkpoint.get('losses', [])
        print(f"Checkpoint loaded from {checkpoint_path}")


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    noise_scheduler,
    num_samples: int = 64,
    image_size: int = 64,
    channels: int = 3,
    device: str = "cpu",
    show_progress: bool = True
) -> torch.Tensor:
    """
    Generate samples using DDPM sampling

    Args:
        model: Trained denoiser model
        noise_scheduler: Noise scheduler with diffusion parameters
        num_samples: Number of samples to generate
        image_size: Size of generated images (assumes square)
        channels: Number of image channels
        device: Device to run on
        show_progress: Whether to show progress bar

    Returns:
        Generated images tensor [num_samples, channels, image_size, image_size]
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    # Reverse diffusion process
    timesteps = reversed(range(noise_scheduler.num_timesteps))
    if show_progress:
        timesteps = tqdm(list(timesteps), desc='Sampling')

    for t in timesteps:
        t_batch = torch.tensor([t] * num_samples, device=device)

        # Predict noise
        predicted_noise = model(x, t_batch)

        # Get alpha values
        alpha = noise_scheduler.alphas[t]
        alpha_cumprod = noise_scheduler.alphas_cumprod[t]
        beta = noise_scheduler.betas[t]

        # Compute x_{t-1}
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        # DDPM sampling equation
        x = (
            1 / torch.sqrt(alpha) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        )

    model.train()
    return x


@torch.no_grad()
def sample_ddpm_with_trajectory(
    model: nn.Module,
    noise_scheduler,
    num_samples: int = 8,
    image_size: int = 64,
    channels: int = 3,
    device: str = "cpu",
    num_steps_to_save: int = 10
) -> list:
    """
    Generate samples and return intermediate steps for visualization

    Args:
        model: Trained denoiser model
        noise_scheduler: Noise scheduler
        num_samples: Number of samples to generate
        image_size: Size of generated images
        channels: Number of image channels
        device: Device to run on
        num_steps_to_save: Number of intermediate steps to save

    Returns:
        List of intermediate images (each is a tensor)
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    # Store intermediate results
    timesteps = list(reversed(range(noise_scheduler.num_timesteps)))
    step_interval = max(1, len(timesteps) // num_steps_to_save)
    intermediate_images = []

    # Reverse diffusion process
    for i, t in enumerate(tqdm(timesteps, desc='Sampling with trajectory')):
        t_batch = torch.tensor([t] * num_samples, device=device)

        # Save intermediate step
        if i % step_interval == 0 or i == len(timesteps) - 1:
            intermediate_images.append(x.cpu().clone())

        # Predict noise
        predicted_noise = model(x, t_batch)

        # Get alpha values
        alpha = noise_scheduler.alphas[t]
        alpha_cumprod = noise_scheduler.alphas_cumprod[t]
        beta = noise_scheduler.betas[t]

        # Compute x_{t-1}
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            1 / torch.sqrt(alpha) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        )

    model.train()
    return intermediate_images


@torch.no_grad()
def visualize_diffusion_forward(
    noise_scheduler,
    image: torch.Tensor,
    timesteps: list,
    device: str = "cpu"
) -> list:
    """
    Visualize the forward diffusion process

    Args:
        noise_scheduler: Noise scheduler
        image: Original image [1, C, H, W] or [C, H, W]
        timesteps: List of timesteps to visualize
        device: Device to run on

    Returns:
        List of noisy images at specified timesteps
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    noisy_images = []

    for t in timesteps:
        t_tensor = torch.tensor([t], device=device)
        noisy_image, _ = noise_scheduler.add_noise(image, t_tensor)
        noisy_images.append(noisy_image.cpu())

    return noisy_images


class EMA:
    """
    Exponential Moving Average for model parameters
    Helps stabilize training and improve sample quality
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
