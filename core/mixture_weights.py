"""
Mixture Weights Computation Module

This module implements attention-based mixture weight computation for Gaussian mixture models.
It uses a learnable attention mechanism to compute weights that determine how much each
Gaussian component contributes to the final mixture.

The attention mechanism allows the model to dynamically focus on different parts of the
encoded history when computing mixture weights, making the model more flexible and expressive.
"""

import math
import pdb

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MixtureWeights(nn.Module):
    """
    Computes mixture weights for Gaussian mixture models using attention mechanism.

    This module uses learnable component embeddings (beta parameters) as queries to attend
    over the encoded history, producing mixture weights that sum to 1.0 via softmax.

    The attention mechanism allows each mixture component to focus on different aspects
    of the historical data, leading to more expressive and adaptive mixture models.

    Args:
        latent_dim: Dimension of the latent/hidden representations
        num_components: Number of mixture components (Gaussians)
        dropout_rate: Dropout rate for regularization (default: 0.1)
    """

    def __init__(
        self, latent_dim: int, num_components: int, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_components = num_components

        # Learnable component embeddings (queries for attention)
        self.component_embeddings = nn.Parameter(
            torch.randn(num_components, latent_dim) * 0.1
        )

        # Attention scaling factor (as in Transformer)
        self.attention_scale = math.sqrt(latent_dim) ** -1

        # Output projection to scalar weights
        self.output_projection = nn.Linear(latent_dim, 1)

        # Initialize parameters

    def _compute_attention_weights(
        self, queries: Tensor, keys: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """
        Compute attention weights using scaled dot-product attention.

        Args:
            queries: Query vectors [num_components, latent_dim]
            keys: Key vectors [batch_size, seq_len, latent_dim]
            attention_mask: Mask for valid positions [batch_size, num_components, seq_len]

        Returns:
            Attention weights [batch_size, num_components, seq_len]
        """
        batch_size = keys.size(0)

        # Expand queries for batch processing
        queries_expanded = queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attention_scores = (
            torch.bmm(queries_expanded, keys.transpose(-2, -1)) * self.attention_scale
        )

        # Apply mask (set masked positions to large negative value)
        masked_scores = attention_scores.masked_fill(~attention_mask, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(masked_scores, dim=-1)

        return attention_weights

    def _aggregate_with_attention(
        self, attention_weights: Tensor, values: Tensor
    ) -> Tensor:
        """
        Aggregate values using attention weights.

        Args:
            attention_weights: Attention weights [batch_size, num_components, seq_len]
            values: Value vectors [batch_size, seq_len, latent_dim]

        Returns:
            Aggregated representations [batch_size, num_components, latent_dim]
        """
        # Weighted aggregation: attention_weights @ values
        aggregated = torch.bmm(attention_weights, values)

        return aggregated

    def _create_attention_mask(self, sequence_mask: Tensor) -> Tensor:
        """
        Create attention mask from sequence mask.

        Args:
            sequence_mask: Binary mask [batch_size, seq_len] where 1 = valid, 0 = invalid

        Returns:
            Attention mask [batch_size, num_components, seq_len]
        """
        batch_size, seq_len = sequence_mask.size()

        # Expand mask for all components
        attention_mask = sequence_mask.unsqueeze(1).expand(
            batch_size, self.num_components, seq_len
        )

        return attention_mask.bool()

    def forward(self, encoded_history: Tensor, sequence_mask: Tensor) -> Tensor:
        """
        Compute mixture weights using attention over encoded history.

        The process:
        1. Use component embeddings as queries to attend over encoded history
        2. Compute attention weights based on similarity
        3. Aggregate history representations using attention weights
        4. Project to scalar weights and apply softmax for normalization

        Args:
            encoded_history: Encoded historical data [batch_size, seq_len, latent_dim]
            sequence_mask: Binary mask for valid observations [batch_size, seq_len]
                          (1 for valid positions, 0 for padded/invalid positions)

        Returns:
            Mixture weights: [batch_size, num_components]
                           Each row sums to 1.0 (valid probability distribution)
        """
        # Validate inputs
        batch_size, seq_len, latent_dim = encoded_history.size()
        assert (
            latent_dim == self.latent_dim
        ), f"Expected latent_dim={self.latent_dim}, got {latent_dim}"
        assert sequence_mask.size() == (batch_size, seq_len), "Mask shape mismatch"

        # Create attention mask for all components
        attention_mask = self._create_attention_mask(sequence_mask)

        # Compute attention weights using component embeddings as queries
        attention_weights = self._compute_attention_weights(
            queries=self.component_embeddings,
            keys=encoded_history,
            attention_mask=attention_mask,
        )

        # Aggregate encoded history using attention weights
        aggregated_features = self._aggregate_with_attention(
            attention_weights, encoded_history
        )

        # Project to scalar weights for each component
        # Shape: [batch_size, num_components, 1] -> [batch_size, num_components]
        raw_weights = self.output_projection(aggregated_features).squeeze(-1)

        # Apply softmax to ensure weights sum to 1.0
        log_mixture_weights = F.log_softmax(raw_weights, dim=-1)

        return log_mixture_weights
