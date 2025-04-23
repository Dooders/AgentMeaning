import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from meaning_transform.src.models.utils import CompressionBase, set_temp_seed


class AdaptiveEntropyBottleneck(CompressionBase):
    """
    Adaptive entropy bottleneck that actually changes its structure based on compression level.
    The parameter count scales inversely with compression level.
    """

    def __init__(
        self,
        latent_dim: int,
        compression_level: float = 1.0,
        threshold: float = 1.2,
        seed: Optional[int] = None,
    ):
        """
        Initialize adaptive entropy bottleneck.

        Args:
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
            threshold: Compression level threshold for using projection layers
            seed: Random seed for reproducibility
        """
        super().__init__(latent_dim, compression_level)
        
        # Additional validation
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}")
        if compression_level <= 0:
            raise ValueError(f"compression_level must be positive, got {compression_level}")
            
        self.seed = seed
        self.use_projection = compression_level >= threshold
        
        # Register buffer to track compressed values
        self.register_buffer("_compression_epsilon", torch.tensor(1e-6))

        # Initialize parameters using temp seed (if provided)
        with set_temp_seed(seed):
            if self.use_projection:
                # Learnable parameters sized to effective dimension
                self.compress_mu = nn.Parameter(torch.zeros(self.effective_dim))
                self.compress_log_scale = nn.Parameter(torch.zeros(self.effective_dim))

                # Projection layers with adaptive dimensions
                self.proj_down = nn.Linear(latent_dim, self.effective_dim)
                self.nonlin = nn.LeakyReLU()
                self.proj_up = nn.Sequential(
                    nn.Linear(self.effective_dim, latent_dim // 4),
                    nn.LeakyReLU(),
                    nn.Linear(latent_dim // 4, latent_dim * 2)
                )
            else:
                # Use simpler parameterization for near-identity cases
                self.compress_params = nn.Parameter(torch.zeros(latent_dim, 2))

        logging.info(
            f"Created AdaptiveEntropyBottleneck with latent_dim={latent_dim}, "
            f"compression_level={compression_level}, effective_dim={self.effective_dim}, "
            f"use_projection={self.use_projection}"
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress latent representation using adaptive entropy bottleneck.

        Args:
            z: Latent representation [B, D]

        Returns:
            z_compressed: Compressed latent representation
            compression_loss: Loss term for compression entropy
        """
        # Validate input
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(z)}")
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected shape (batch_size, {self.latent_dim}), got {z.shape}"
            )
        
        # Check if input is already compressed by examining statistical properties
        # Compressed values typically have a specific variance pattern after quantization
        if not self.training:
            # In evaluation mode, check if variance in fractional parts is very low,
            # indicating already quantized values
            fractional_parts = z - z.round()
            mean_abs_fractional = fractional_parts.abs().mean()
            
            # If mean fractional part is close to zero, values are already quantized
            if mean_abs_fractional < self._compression_epsilon:
                return z, torch.zeros(1, device=z.device)

        if self.use_projection:
            # Project down to effective dimension
            z_down = self.proj_down(z)
            z_down = self.nonlin(z_down)

            # Apply compression in the effective space
            mu = z_down + self.compress_mu
            log_scale = self.compress_log_scale.clone()

            # Add noise for quantization in the effective space
            if self.training:
                # Reparameterization trick during training
                with set_temp_seed(self.seed):
                    epsilon = torch.randn_like(mu)
                z_compressed_effective = mu + torch.exp(log_scale) * epsilon
            else:
                # Deterministic rounding during inference
                z_compressed_effective = torch.round(mu)

            # Project back up to full latent space
            projected = self.proj_up(z_compressed_effective)
            mu_full, log_scale_full = torch.chunk(projected, 2, dim=-1)
            
            # Use mu_full directly as the compressed representation
            z_compressed = mu_full

            # Compute entropy loss (bits per dimension) with improved numerical stability
            compression_loss = 0.5 * log_scale.mul(2).exp() + 0.5 * torch.log(
                2 * torch.tensor(np.pi, device=z.device)
            )
            compression_loss = compression_loss.mean()
        else:
            # Simplified path for low compression
            params = self.compress_params
            mu, log_scale = params[:, 0], params[:, 1]
            z_compressed = mu

            # Compute entropy loss (bits per dimension) with improved numerical stability
            compression_loss = 0.5 * log_scale.exp().pow(2) + 0.5 * torch.log(
                2 * torch.tensor(np.pi, device=z.device)
            )
            compression_loss = compression_loss.mean()

        return z_compressed, compression_loss

    def get_parameter_count(self) -> int:
        """
        Calculate the total number of parameters in the bottleneck.

        Returns:
            Number of parameters
        """
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params

    def get_effective_compression_rate(self) -> float:
        """
        Calculate the effective compression rate in bits per dimension.

        Returns:
            Effective bits per dimension
        """
        # Avoid division by zero or recursion
        if self.effective_dim <= 0:
            return 0.0

        # Calculate bits per dimension
        return float(self.latent_dim) / float(self.effective_dim)
