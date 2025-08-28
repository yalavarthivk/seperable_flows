__all__ = [
    "Moses",
]


from typing import Final

import torch
from networkx import reverse
from torch import Tensor, nn

from .distributions.base_distribution import MultiGaussian
from .distributions.transforms import RationalLinearSplineFlow
from .encoders.transformer import TimeSeriesEncoder as Encoder
from .mixture_weights import MixtureWeights
from .utils.metrics import compute_mnll, compute_njnll


class Moses:
    r"""Implements a MOSES"""

    def __init__(
        self,
        n_inputs: int = 5,
        n_heads: int = 2,
        num_components=2,
        latent_dim: int = 128,
        num_flow_layers: int = 3,
        num_bins: int = 16,
        bounds: float = 20.0,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        # constants
        self.n_inputs = n_inputs
        self.n_heads = (
            n_heads  # number of attn heads to compute encoder, cov and mix weights
        )
        self.num_flow_layers = num_flow_layers
        self.num_components = num_components
        self.num_bins = num_bins
        self.bounds = bounds
        self.latent_dim = latent_dim
        self.conditioning = None
        self.mixture_logits = None
        self.full_cond = None
        self.history_cond = None
        self.log_mix_wts = None
        # submodules

        # spline functions
        self.flow = nn.ModuleList()
        for _ in range(self.num_flow_layers):
            self.flow.append(RationalLinearSplineFlow(self.latent_dim))

        # base distribution Gaussian Process
        self.base = MultiGaussian(latent_dim=latent_dim)

        # Mixture components
        self.mixture_weights = MixtureWeights(
            latent_dim=latent_dim, num_components=num_components
        )

        # encoding the conditioning
        self.encoder = Encoder(n_inputs, latent_dim, num_components, n_heads)

    def distribution(self, tx, cx, mx, x, tq, cq, mq):
        r"""encode the conditioning input"""
        self.full_cond, self.history_cond = self.encoder(tx, cx, mx, x, tq, cq, mq)  #
        self.log_mix_wts = self.mixture_weights(self.history_cond, mx)
        self.base(mq, self.full_cond)
        for spline in self.flow:
            spline.forward_(self.full_cond)

    def flow_forward(self, y):
        r"""Compute the forward operation on flows"""
        ldj = torch.zeros_like(y)
        for spline in self.flow:
            y, ldj_flow = spline.forward_(y)
            ldj = ldj + ldj_flow
        return y, ldj

    def flow_inverse(self, z):
        """compute the inverse operation on flows"""
        ldj = torch.zeros_like(z)
        for spline in reversed(self.flow):
            z, ldj_flow = spline.inverse_(z)
            ldj = ldj + ldj_flow
        return z, ldj

    def njnll(self, y: Tensor, mq) -> Tensor:
        """Compute the normalized joint negative log likelihood of the input."""
        y_ = y.unsqueeze(1).repeat(1, self.num_components, 1)

        z, ldj = self.flow_forward(y_)  # z \in [B, K, D], ldj \in [B, D]

        likelihood = compute_njnll(
            z, ldj, self.base.mean, self.base.cov, self.log_mix_wts, mq
        )  # B

        return likelihood.mean()

    def mnll(self, y: Tensor, mq) -> Tensor:
        """Compute the marginal negative log likelihood of the input."""
        # y \in [B, K]; mq \in [B, K]
        y_ = y.unsqueeze(1).repeat(1, self.num_components, 1)  # y \in [B, D, K]

        ldj = torch.zeros_like(y_)

        z, ldj = self.flow_forward(y)  # z \in [B, D, K], ldj \in [B, D, K]

        likelihood = compute_mnll(
            z, self.base.mean, self.base.cov, ldj, mq, log_mix_wts
        )  # B, K

        return likelihood.mean()  # average over all targets

    def joint_sample(self, mq: Tensor, nsamples: int = 1000) -> Tensor:
        r"""Sample from the model."""

        # sample from the base distribution
        z = self.base.sample(mq, nsamples)

        x = self.flow_inverse(z)

        indices = self.sample_indices()

        x = x[:, :, indices, :]  # TODO need to do it in correct manner
        return x

    def marginal_sample(self, mq: Tensor, nsamples: int = 1000) -> Tensor:
        z = self.base.marginal_sample(mq, nsamples)
        x = self.flow_inverse(z)
        indices = self.sample_indices()
        x = x[:, :, indices, :]
        return x

    def sample_indices(self) -> Tensor:  # shape: N
        """Return indices of the mixture."""

        mix_weights = torch.exp(self.log_mix_wts)

        indices = torch.multinomial(mix_weights, self.num_components, replacement=True)
        return indices
