import math
import pdb
from typing import Any, Optional

import numpy as np
import properscoring as ps
import torch
from scipy import stats
from scipy.spatial import cKDTree as KDTree
from torch import Tensor, nn


def compute_energy_score(y, yhat, mask, beta=1):
    """
    y: [batch, dim]
    yhat: [batch, nsamples, dim]
    mask: optional, [batch, dim]
    """
    _, nsamples, _ = yhat.shape

    valid_batch = mask.sum(dim=1) > 0
    y = y[valid_batch] * mask[valid_batch]
    yhat = yhat[valid_batch] * mask[valid_batch, None, :]

    # First term: mean over samples
    diff = torch.cdist(yhat, y[:, None, :], p=2)  # [batch, nsamples, 1]
    first_term = diff.pow(beta).mean(dim=1).squeeze(-1)  # mean over nsamples

    # Second term: pairwise distances, exclude diagonal
    pairwise = torch.cdist(yhat, yhat, p=2)
    diag = torch.diagonal(pairwise, dim1=1, dim2=2)
    pairwise = pairwise - torch.diag_embed(diag)
    second_term = -pairwise.pow(beta).sum(dim=(1, 2)) / (2 * nsamples**2)

    energy = first_term + second_term
    return energy  # per batch


def compute_mse(yhat, y, mask):
    squared_error = (y - yhat) ** 2
    sse = squared_error * mask
    mse = sse.sum() / mask.sum()
    return mse


def compute_crps(yhat, y, mask):
    y_expanded = y.unsqueeze(1)  # [B, 1, K]

    term1 = torch.mean(torch.abs(y_expanded - yhat), dim=1)  # [B, K]

    # Compute pairwise differences between ensemble members
    yhat1 = yhat.unsqueeze(2)  # [B, S, 1, K]
    yhat2 = yhat.unsqueeze(1)  # [B, 1, S, K]
    term2 = 0.5 * torch.mean(torch.abs(yhat1 - yhat2), dim=(1, 2))  # [B, K]

    crps = term1 - term2

    # Apply mask
    return crps * mask


def multivariate_normal(z, mu, sigma, mq):
    r"""
    Log-density of a multivariate Gaussian using Cholesky decomposition.

    Formula:
        log p(x) = - (K / 2) * log(2π)
                - 0.5 * || L^{-1} (x - μ) ||^2
                - sum_i log L_ii

    Where:
        - x : data vector
        - μ : mean vector
        - L : lower-triangular Cholesky factor of covariance Σ (Σ = L @ L.T)
        - K : dimension of x

    """
    L = torch.linalg.cholesky(sigma)  # [..., D, D]
    diff = z - mu  # [..., D]
    solve_l_diff = torch.linalg.solve_triangular(
        L, diff.unsqueeze(-1), upper=False
    ).squeeze(-1)
    quad = (solve_l_diff**2).sum(-1)
    ldj = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    D = mq.sum(-1)
    log_prob = -0.5 * (D * math.log(2 * math.pi) + quad) - ldj
    return -log_prob


def compute_njnll(
    z: Tensor,
    mu: Tensor,
    sigma: Tensor,
    ldj: Tensor,
    mq: Tensor,
    log_mix_wts: Tensor,
) -> Tensor:
    r"""Compute njNLL of given values"""
    # mu is zero where there is mask and sigma is 1 for the diagonal where there is mask
    log_prob = multivariate_normal(z, mu, sigma, mq)
    comp_log_prob = log_prob - ldj
    total_nll = -torch.logsumexp(log_mix_wts - comp_log_prob, -1)
    return total_nll


def compute_mnll(,
    z: Tensor,
    mu: Tensor,
    sigma: Tensor,
    ldj: Tensor,
    mq: Tensor,
    log_mix_wts: Tensor,
) -> Tensor:
    r"""Compute NLL of latent distribution, Gaussian"""

    base_prob = base_dist(mu, z, sigma) * mq  # loglikelihood of the base distribution
    comp_loss = base_prob - ldj  # loglikelihood for each component

    total_loss = -torch.logsumexp(
        log_mix_wts - comp_loss, -1
    )  # logsumexp trick p(y) = w1p_1(y) + ... + wDp_D(y)
    return total_loss


def compute_robust_mean(yhat):
    pass

def wasserstein_distance()

class compute_losses(nn.Module):
    r"""compute losses"""

    def __init__(self):
        super(compute_losses).__init__()
        self.nl_base_marginal_loss = nn.GaussianNLLLoss(full=True, reduction="none")

    def forward(self, model, y, mq):
        y_ = y.unsqueeze(1).repeat(1, model.num_components, 1)

        z, ldj = model.flow_forward(y_)  # z \in [B, K, D], ldj \in [B, D]
        mu = model.base_mean
        sigma = model.base_cov
        mix_wts = model.mix_wts
        log_mix_wts = torch.log(mix_wts)  # log of mixture weights
        mnll = compute_mnll(y_, z, mu, sigma, ldj, log_mix_wts, mq)
        njnll = compute_njnll(y_, z, mu, sigma, ldj, log_mix_wts, mq)
        return mnll, njnll


class AdditionalLossMetrics:
    """Container for computed loss metrics."""

    mse: Optional[float] = None
    robust_mse: Optional[float] = None
    crps: Optional[float] = None
    energy_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert metrics to dictionary format."""
        return {
            "mse": self.mse,
            "robust_mse": self.robust_mse,
            "crps": self.crps,
            "energy_score": self.energy_score,
        }


class AdditionalLossesComputer:
    """
    Computes additional loss metrics for probabilistic models.

    This class calculates various loss metrics including MSE, robust MSE,
    CRPS (Continuous Ranked Probability Score), and Energy Score.
    """

    def __init__(self):
        """Initialize the loss computer with empty metrics."""
        self.metrics = AdditionalLossMetrics()

    def compute_losses(
        self, model: Any, y: Tensor, mq: Tensor, n_samples: int = 1000
    ) -> AdditionalLossMetrics:
        """
        Compute all loss metrics for the given model predictions.

        Args:
            model: Probabilistic model with a samples() method
            y: Ground truth targets [B, K]
            mq: Model quantiles or other reference [B, K]
            n_samples: Number of samples to draw from the model

        Returns:
            LossMetrics: Container with all computed metrics

        Raises:
            ValueError: If inputs have incompatible shapes
            AttributeError: If model doesn't have required methods
        """
        # Validate inputs
        if not hasattr(model, "samples"):
            raise AttributeError("Model must have a 'samples' method")

        if y.shape != mq.shape:
            raise ValueError(f"Shape mismatch: y.shape={y.shape}, mq.shape={mq.shape}")

        # Generate samples from model: [B, n_samples, K]
        yhat = model.samples(nsamples=n_samples)

        # Compute different estimators
        yhat_mean = yhat.mean(dim=1)  # Standard mean
        yhat_robust_mean = compute_robust_mean(yhat)  # Robust estimator

        # Calculate all metrics
        self.metrics.mse = compute_mse(yhat_mean, y, mq)
        self.metrics.robust_mse = compute_mse(yhat_robust_mean, y, mq)
        self.metrics.crps = compute_crps(yhat, y, mq)
        self.metrics.energy_score = compute_energy_score(yhat, y, mq)

        return self.metrics

def WassersteinMetric(model, mq, nsamples=1000):
    marginal_y = model.marginal_samples(nsamples=nsamples) # samples drawn from the marginals
    marginal_y_numerical = model.joint_samples(nsamples=nsamples) # samples drawn from the joints
    marginal_y_sorted = marginal_y.sort(-1)[0] # sort the samples
    marginal_y_numerical_sorted = marginal_y_numerical.sort(-1)[0]
    dist = (marginal_y_sorted - marginal_y_numerical_sorted) ** 2
    dist_mean = (dist.mean(-1)) ** 0.5
    return dist_mean*mq