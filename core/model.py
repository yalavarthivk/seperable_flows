__all__ = [
    "MarginalizableNormalizingFlow",
]


from typing import Final

import torch
from torch import Tensor, nn

from core.distributions.distributions import Distribution
from core.distributions.base_distribution import MultiGaussian
from core.transforms.lr_splines import SplineFlow
from core.mixture_weights import MixtureWeights
from core.encoders import Encoder

from utils.metrics import compute_njnll, compute_mnll, crps, energy_score


class Moses(Distribution[Tensor, Tensor]):
    r"""Implements a MOSES"""

    NUM_COMPONENTS: Final[int]
    """Number of mixture components."""
    NUM_FLOW_LAYERS: Final[int]
    """Number of rational linear spline layers."""
    NUM_BINS: Final[int]
    """Number of bins in the rational linear splines."""
    BOUNDS: Final[float]
    """Tail bound of the rational linear splines."""

    # non-permanent buffers
    sample_indices: Tensor
    """Indices of the mixture components."""
    latents: Tensor
    """Latent state of the model."""
    log_probs: Tensor
    """NLL of the Gaussian base distribution."""
    samples: Tensor
    """Samples from the model."""

    # submodules
    flow: SplineFlow
    base: MultiHeadGaussian
    mixtures: MixtureWeights

    def __init__(
        self,
        n_inputs: int = 5,
        n_heads: int = 2,
        num_components=2,
        latent_dim: int = 128,
        num_flow_layers: int = 3,
        training_marginals: bool = False,
        num_bins: int = 16,
        bounds: float = 20.0,
        device: torch.device = "cuda",
    ) -> None:
        super().__init__()
        # constants
        self.n_inputs = n_inputs
        self.n_heads = n_heads  # number of attention heads used in computing encoder, cov and mix weights
        self.num_flow_layers = num_flow_layers
        self.num_components = num_components
        self.num_bins = num_bins
        self.bounds = bounds
        self.latent_dim = latent_dim
        self.conditioning = None
        self.mixture_logits = None
        self.full_cond = None
        self.history_cond = None
        self.mix_wts = None
        # submodules
        self.flow = SplineFlow(
            num_components=num_components,
            latent_dim=latent_dim,
            num_flow_layers=num_flow_layers,
            num_bins=num_bins,
            bounds=bounds,
        )
        self.base = MultiGaussian(
            latent_dim=latent_dim
        )  # base distribution a multivariate gaussian
        self.mixture_weights = MixtureWeights(
            latent_dim=latent_dim, num_components=num_components
        )
        self.encoder = Encoder(n_inputs, latent_dim, num_components, n_heads)

    def get_mixture_logits(self) -> Tensor:
        r"""Compute mixture logits from mixture parameters."""
        logits = self.mixture_params.log_softmax(dim=0)
        self.mixture_logits = logits
        return logits

    def encode_input(self, tx, cx, mx, x, tq, cq, mq) -> Tensor:
        r"""encode the conditioning input"""
        self.full_cond, self.history_cond = self.encoder(tx, cx, mx, x, tq, cq, mq)
        self.mix_wts = self.mixture_weights(self.history_cond, mx)

    def njnll(self, y: Tensor, mq) -> Tensor:
        """Compute the normalized joint negative log likelihood of the input."""

        z, ldj = self.flow_forward(
            y, mq, self.full_cond
        )  # z \in [B, K, D], ldj \in [B, D]

        mu, sigma = self.base(mq, self.full_cond)

        likelihood = compute_njnll(z, ldj, mu, sigma, self.mix_wts, mq)  # B

        return likelihood.mean()

    def mnll(self, y: Tensor, mq) -> Tensor:
        """Compute the marginal negative log likelihood of the input."""

        z, ldj = self.flow_forward(
            y, mq, self.full_cond
        )  # z \in [B, K, D], ldj \in [B, D]

        mu, sigma = self.base(mq, self.full_cond)

        likelihood = compute_mnll(z, ldj, mu, sigma, self.mix_wts, mq)  # B

        return likelihood  # already averaged over all the queries

    def sample(self, mq: Tensor, nsamples: int) -> Tensor:
        r"""Sample from the model."""

        # sample from the base distribution
        z = self.base.sample(self.full_cond, mq, nsamples)

        x = self.flow.inverse(z, self.full_cond, nsamples)

        indices = self.sample_indices()

        x = x[:, :, indices, :]
        return x

    def sample_indices(self) -> Tensor:  # shape: N
        """Return indices of the mixture."""

        indices = torch.multinomial(
            self.mix_weights, self.num_components, replacement=True
        )
        return indices

    def compute_mean(self, mq: Tensor) -> Tensor:
        """Compute the mean of the prediction

        Args:
            mq (Tensor): [B, K] mask

        Returns:
            Tensor: mean of the distirbution [B, K]
        """
        samples = self.sample(mq, nsamples=1000)  # [B, nsamples, K]
        mean_ = samples.mean(2)  # [B, K]
        return mean_

    def mse(self, y: Tensor, mq: Tensor) -> Tensor:
        """Compute the mean squared error between the distribution mean and the ground truth

        Args:
            y (Tensor): ground truth value
            mq (Tensor): mask

        Returns:
            Tensor: mse
        """
        yhat = self.compute_mean(mq)
        squared_error = mq * ((y - yhat) ** 2)
        mse = squared_error.sum() / mq.sum()
        return mse
