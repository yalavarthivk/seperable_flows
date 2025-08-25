import math
import torch
from torch import nn, Tensor


class MultiGaussian(nn.Module):
    r"""Compute the parameters of a mixture of Gaussians."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.mean_layer = nn.Linear(latent_dim, 1)
        self.query_layer = nn.Linear(latent_dim, latent_dim)
        self.latent_dim = latent_dim

    def cov(self, x: Tensor, mask: Tensor) -> Tensor:
        r"""Compute covariance matrix with attention-style mechanism.

        Args:
            x: Tensor of shape (batch_size, num_components, num_queries, hidden_dim)
            mask: Binary tensor (batch_size, num_components, num_queries)

        Returns:
            cov: Tensor of shape (batch_size, num_components, num_queries, num_queries)
        """
        b, c, q, _ = x.shape
        queries = self.query_layer(x)

        # Attention-style similarity scores
        scores = torch.matmul(queries, queries.transpose(-2, -1)) / math.sqrt(
            self.latent_dim
        )

        # Identity matrix broadcasted across batch & components
        id_matrix = torch.eye(q, device=x.device).expand(b, c, -1, -1)

        # Mask into covariance
        attn_mask = mask.unsqueeze(-1) @ mask.unsqueeze(-2)  # (b, c, q, q)
        cov = attn_mask * (scores + id_matrix) + (1 - attn_mask) * id_matrix
        return cov

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        r"""Output mean and covariance.

        Args:
            x: Tensor of shape (batch_size, num_components, num_queries, hidden_dim)
            mask: Binary tensor (batch_size, num_components, num_queries)

        Returns:
            mean: (batch_size, num_components, num_queries)
            cov:  (batch_size, num_components, num_queries, num_queries)
        """
        mean = self.mean_layer(x).squeeze(-1)
        cov = self.cov(x, mask)
        return mean, cov
