"""Compute mixture weights"""

import math

import torch
from torch import nn, Tensor


class MixtureWeights(nn.Module):
    r"""Compute the parameters of a mixture of Gaussians."""

    def __init__(self, latent_dim: int, num_components: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.beta = nn.Parameter(torch.rand(num_components))
        self.scale = math.sqrt(latent_dim) ** -1
        self.out_layer = nn.Linear(latent_dim, 1)

    def attention(self, query, key, values, mask):
        r"""
        Compute mix weights using attention
        """
        score = torch.bmm(query, key.transpose(-1, -2)) * self.scale
        masked_score = score.masked_fill(mask, 1e-8)
        normalized_scores = nn.softmax(masked_score, -2)
        output = nn.relu()(torch.bmm(normalized_scores, values))
        attn_out = self.out_layer(output)
        return attn_out.squeeze(-1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        r"""Compute mixture weights

        Args:
            x: encoded information of only history; (batch_size, nobs, hidden_dim)
            mask: mask for observations: (bathc_size, nobs)

        Returns:
            mixture weights: Tensor of shape (batch_size, num_components)
        """
        attn_mask = mask[:, None, :].repeat(self.num_components)
        mw = self.attention(self.beta, x, x, mask)  # mixture weights [B, D]

        return nn.Softmax(-1)(mw)  # softmax to make it convex sum
