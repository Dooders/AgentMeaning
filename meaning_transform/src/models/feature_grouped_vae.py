import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from meaning_transform.src.models.adaptive_entropy_bottleneck import (
    AdaptiveEntropyBottleneck,
)
from meaning_transform.src.models.decoder import Decoder
from meaning_transform.src.models.encoder import Encoder
from meaning_transform.src.models.utils import BaseModelIO


class FeatureGroupedVAE(nn.Module, BaseModelIO):
    """
    VAE model that applies different compression rates to different feature groups based on importance.
    This allows for better semantic preservation of high-importance features.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        feature_groups: Optional[Dict[str, Tuple[int, int, float]]] = None,
        base_compression_level: float = 1.0,
        use_batch_norm: bool = True,
        min_group_dim: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Initialize feature-grouped VAE model.

        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            feature_groups: Dictionary mapping group names to (start_idx, end_idx, compression_level)
                            If None, equal groups with equal compression will be used
            base_compression_level: Base compression level to apply to all groups
            use_batch_norm: Whether to use batch normalization
            min_group_dim: Minimum dimension allowed for any feature group
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.base_compression_level = base_compression_level
        self.seed = seed
        self.use_batch_norm = use_batch_norm
        self.min_group_dim = max(1, min_group_dim)  # Ensure minimum is at least 1

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Default feature groups if not provided
        if feature_groups is None:
            # Equal splits with equal compression
            group_size = input_dim // 3
            feature_groups = {
                "group1": (0, group_size, 1.0),
                "group2": (group_size, 2 * group_size, 1.0),
                "group3": (2 * group_size, input_dim, 1.0),
            }

        self.feature_groups = feature_groups

        # Create encoder and decoder
        self.encoder = Encoder(input_dim, latent_dim, use_batch_norm=use_batch_norm)
        self.decoder = Decoder(latent_dim, input_dim, use_batch_norm=use_batch_norm)

        # Create shared compressor
        self.shared_compressor = nn.Module()
        self.shared_compressor.mu_network = nn.Linear(latent_dim, latent_dim)
        self.shared_compressor.scale_network = nn.Linear(latent_dim, latent_dim)

        # Create separate bottlenecks for each feature group
        self.group_params = nn.ParameterDict()
        self.group_latent_dims = {}

        # Determine latent dimension allocation per group
        total_features = sum(end - start for start, end, _ in feature_groups.values())

        latent_start_idx = 0
        group_names = list(feature_groups.keys())

        # First pass: calculate intended dimensions
        intended_dims = {}
        for name, (start_idx, end_idx, compression) in feature_groups.items():
            feature_count = end_idx - start_idx
            intended_dims[name] = max(
                self.min_group_dim, int(latent_dim * feature_count / total_features)
            )

        # Adjust to ensure total matches latent_dim exactly
        total_allocated = sum(intended_dims.values())
        remaining = latent_dim - total_allocated

        # Distribute remaining dimensions to groups proportionally
        if remaining != 0:
            # Sort groups by compression value (lower compression = higher importance)
            sorted_groups = sorted(
                intended_dims.items(),
                key=lambda x: feature_groups[x[0]][2],  # Sort by compression value
            )
            
            if remaining > 0:
                # Add extra dimensions to important groups first
                for i in range(remaining):
                    group_name = sorted_groups[i % len(sorted_groups)][0]
                    intended_dims[group_name] += 1
            else:
                # Take dimensions from less important groups first (reverse order)
                # Sort by descending importance (higher compression = lower importance)
                sorted_groups.reverse()
                
                # First check if there are any groups above minimum that can be reduced
                reducible_groups = [
                    name for name, dim in intended_dims.items() 
                    if dim > self.min_group_dim
                ]
                
                if not reducible_groups:
                    logging.warning(
                        f"Cannot reduce dimensions: all groups are at minimum {self.min_group_dim} dimensions. "
                        f"Requesting {abs(remaining)} fewer dimensions than available."
                    )
                    # Adjust latent_dim to match what we can actually allocate
                    self.latent_dim = sum(intended_dims.values())
                    logging.info(f"Adjusted latent_dim to {self.latent_dim}")
                else:
                    # Proceed with dimension reduction
                    for i in range(abs(remaining)):
                        group_name = sorted_groups[i % len(sorted_groups)][0]
                        # Ensure we don't reduce any group below minimum threshold
                        if intended_dims[group_name] > self.min_group_dim:
                            intended_dims[group_name] -= 1
                        else:
                            # If we can't take from this group, find any group above minimum
                            found_reducible = False
                            for next_group in sorted_groups:
                                group_name = next_group[0]
                                if intended_dims[group_name] > self.min_group_dim:
                                    intended_dims[group_name] -= 1
                                    found_reducible = True
                                    break
                                    
                            if not found_reducible:
                                logging.warning(
                                    f"Cannot reduce more dimensions: all groups at minimum {self.min_group_dim}. "
                                    f"Stopping after reducing {i} of {abs(remaining)} dimensions."
                                )
                                # Adjust latent_dim to match what we can actually allocate
                                self.latent_dim = sum(intended_dims.values())
                                logging.info(f"Adjusted latent_dim to {self.latent_dim}")
                                break

        # Second pass: create bottlenecks with corrected dimensions
        for i, name in enumerate(group_names):
            start_idx, end_idx, compression = feature_groups[name]
            group_latent_dim = intended_dims[name]

            # Apply group-specific compression
            effective_compression = compression * base_compression_level

            # Store latent dimension range for this group
            self.group_latent_dims[name] = (
                latent_start_idx,
                latent_start_idx + group_latent_dim,
            )

            # Create parameter for this group
            self.group_params[name] = nn.Parameter(torch.zeros(group_latent_dim, 2))

            latent_start_idx += group_latent_dim

            logging.info(
                f"Feature group '{name}': features [{start_idx}:{end_idx}], "
                f"latent dim {group_latent_dim}, compression {effective_compression}x"
            )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from latent distribution."""
        std = torch.exp(0.5 * log_var)

        if self.training:
            # Always use the set_temp_seed context manager for consistency
            from meaning_transform.src.models.utils import set_temp_seed
            with set_temp_seed(self.seed):
                eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        else:
            # During evaluation, just use the mean for deterministic results
            return mu

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through the feature-grouped VAE model."""
        # Validate input
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected shape (batch_size, {self.input_dim}), got {x.shape}")
        if x.size(0) < 1:
            raise ValueError(f"Batch size must be at least 1, got {x.size(0)}")
            
        # Check for NaN or infinity values
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains infinity values")
        
        # Encode input to latent space
        mu, log_var = self.encoder(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)

        # Apply feature-specific compression
        z_compressed = torch.zeros_like(z)
        compression_loss = 0.0

        for name, (start_idx, end_idx) in self.group_latent_dims.items():
            # Get latent vector segment for this group
            z_group = z[:, start_idx:end_idx]

            # Apply shared compressor
            mu_group = self.shared_compressor.mu_network(z_group)
            log_scale_group = self.shared_compressor.scale_network(z_group)

            # Add noise for quantization in the effective space
            if self.training:
                # Reparameterization trick during training
                with set_temp_seed(self.seed):
                    epsilon = torch.randn_like(mu_group)
                z_group_compressed = mu_group + torch.exp(log_scale_group) * epsilon
            else:
                # Deterministic rounding during inference
                z_group_compressed = torch.round(mu_group)

            # Store compressed representation
            z_compressed[:, start_idx:end_idx] = z_group_compressed

            # Accumulate compression loss
            compression_loss += 0.5 * log_scale_group.mul(2).exp() + 0.5 * torch.log(
                2 * torch.tensor(torch.pi, device=z.device)
            )

        # Average compression loss across groups
        compression_loss = compression_loss.mean()

        # Decode compressed representation
        x_reconstructed = self.decoder(z_compressed)

        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size

        # Return all tensors and loss components
        return {
            "reconstruction": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_compressed": z_compressed,
            "kl_loss": kl_loss,
            "compression_loss": compression_loss,
            "feature_group_dims": self.group_latent_dims,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input tensor to compressed latent space."""
        # Check for NaN or infinity values
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains infinity values")
        
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        # Apply feature-specific compression
        z_compressed = torch.zeros_like(z)

        for name, (start_idx, end_idx) in self.group_latent_dims.items():
            # Get latent vector segment for this group
            z_group = z[:, start_idx:end_idx]

            # Apply shared compressor
            mu_group = self.shared_compressor.mu_network(z_group)
            log_scale_group = self.shared_compressor.scale_network(z_group)

            # Add noise for quantization in the effective space
            if self.training:
                # Reparameterization trick during training
                with set_temp_seed(self.seed):
                    epsilon = torch.randn_like(mu_group)
                z_group_compressed = mu_group + torch.exp(log_scale_group) * epsilon
            else:
                # Deterministic rounding during inference
                z_group_compressed = torch.round(mu_group)

            # Store compressed representation
            z_compressed[:, start_idx:end_idx] = z_group_compressed

        return z_compressed

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent representation to reconstruction.

        Args:
            z: Latent representation

        Returns:
            reconstruction: Reconstructed output
        """
        # Check for NaN or infinity values
        if torch.isnan(z).any():
            raise ValueError("Input tensor contains NaN values")
        if torch.isinf(z).any():
            raise ValueError("Input tensor contains infinity values")
        
        return self.decoder(z)

    def get_compression_rate(self) -> Dict[str, float]:
        """Get the effective compression rate for each feature group."""
        rates = {}

        for name, (start_idx, end_idx) in self.group_latent_dims.items():
            group_latent_dim = end_idx - start_idx
            rates[name] = group_latent_dim / self.shared_compressor.mu_network.out_features

        # Also compute overall rate - weighted average based on feature counts
        total_input_features = sum(
            end - start for start, end, _ in self.feature_groups.values()
        )
        weighted_compression = 0.0

        for name, (feature_start, feature_end, _) in self.feature_groups.items():
            feature_count = feature_end - feature_start
            weight = feature_count / total_input_features
            group_rate = rates[name]
            weighted_compression += weight * group_rate

        rates["overall"] = weighted_compression

        return rates

    def get_feature_group_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed analysis of each feature group."""
        analysis = {}

        for name in self.feature_groups:
            start_idx, end_idx, compression = self.feature_groups[name]
            latent_start, latent_end = self.group_latent_dims[name]

            analysis[name] = {
                "feature_range": (start_idx, end_idx),
                "feature_count": end_idx - start_idx,
                "latent_range": (latent_start, latent_end),
                "latent_dim": latent_end - latent_start,
                "effective_dim": self.shared_compressor.mu_network.out_features,
                "compression": compression,
                "base_compression": self.base_compression_level,
                "overall_compression": compression * self.base_compression_level,
                "importance": 1.0
                / compression,  # Inverse relationship with compression
            }

        return analysis

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        config = super().get_config()
        # Add FeatureGroupedVAE specific configuration
        config.update({
            "feature_groups": self.feature_groups,
            "base_compression_level": self.base_compression_level,
            "use_batch_norm": self.use_batch_norm,
        })
        return config

    def save(self, filepath: str) -> None:
        """Save model to file."""
        super().save(filepath)

    def load(self, filepath: str) -> None:
        """Load model from file."""
        super().load(filepath)
