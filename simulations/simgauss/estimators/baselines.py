"""Non-amortized baselines."""

import numpy as np


def population_mle(x_flat, n_contexts):
    """One Gaussian for all data, broadcast to every context."""
    mu = x_flat.mean(axis=0)
    cov = np.cov(x_flat, rowvar=False, bias=True)
    p = x_flat.shape[1]
    return np.tile(mu, (n_contexts, 1)), np.tile(cov.reshape(1, p, p), (n_contexts, 1, 1))


def grouped_mle(x_train):
    """One Gaussian per condition.

    Singular whenever replicates do not exceed the dimension, and degenerate at
    one replicate, where there is no within-condition scatter and the covariance
    is identically zero. We report that collapse rather than regularizing it
    away: it is the reason amortization is needed.
    """
    n_contexts, m, p = x_train.shape
    mus = x_train.mean(axis=1)
    covs = np.zeros((n_contexts, p, p))
    for c in range(n_contexts):
        covs[c] = np.cov(x_train[c], rowvar=False, bias=True).reshape(p, p)
    return mus, covs
