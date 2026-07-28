"""Analytic checks on the matrix routines.

These are cheap and they guard the class of bug that is hardest to notice: a
projection or inversion that silently returns something plausible but wrong.
Every recovery number in the paper flows through these functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simgauss.linalg import (cov_to_corr, cov_to_partial, is_pd,  # noqa: E402
                             min_eigenvalue, to_pd)
from simgauss.generators import generate  # noqa: E402


def known_precisions(n=5, p=8, degree=2, seed=0):
    """Diagonally dominant precision matrices with a fixed-degree support."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, p, p))
    for s in range(n):
        T = np.zeros((p, p))
        for i in range(p):
            for j in range(i + 1, min(i + 1 + degree, p)):
                T[i, j] = T[j, i] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
        T[np.arange(p), np.arange(p)] = 0.1 + np.abs(T).sum(1)
        out[s] = T
    return out


def test_partial_correlations_recover_exactly():
    """Inverting the correlation matrix must return the true partial correlations.

    This is the identity the precision representation rests on: partial
    correlations are scale free, so no marginal variances are needed.
    """
    theta = known_precisions()
    sigma = np.linalg.inv(theta)
    _, recovered = cov_to_partial(cov_to_corr(sigma))

    d = np.sqrt(np.diagonal(theta, axis1=-2, axis2=-1))
    expected = -theta / (d[:, :, None] * d[:, None, :])
    idx = np.arange(theta.shape[-1])
    expected[:, idx, idx] = 1.0

    assert np.abs(recovered - expected).max() < 1e-10


def test_diagonal_dominance_gives_valid_gaussians():
    assert is_pd(known_precisions()).all()


def test_projection_repairs_an_indefinite_matrix():
    bad = np.array([[[1.0, 0.99, 0.99], [0.99, 1.0, -0.99], [0.99, -0.99, 1.0]]])
    assert not is_pd(bad)[0]
    fixed, moved = to_pd(bad)
    assert is_pd(fixed)[0]
    assert moved[0] > 0


def test_projection_is_a_noop_on_valid_input():
    sigma = np.linalg.inv(known_precisions())
    fixed, moved = to_pd(sigma)
    assert moved.max() < 1e-8
    assert np.abs(fixed - sigma).max() < 1e-8


def test_generator_support_matches_nonzero_precision():
    """The recorded support must be exactly where the precision is nonzero,
    otherwise edge recovery is scored against the wrong answer."""
    truth = generate(n_contexts=4, samples_per_context=2, p=8, degree=2,
                     context_dim=3, seed=0)
    off = ~np.eye(8, dtype=bool)
    nonzero = (np.abs(truth["precisions"]).max(axis=0) > 0) & off
    assert (nonzero == (truth["support"] & off)).all()


@pytest.mark.parametrize("p,degree", [(8, 2), (16, 2), (16, 3)])
def test_fixed_degree_keeps_partial_correlations_detectable(p, degree):
    """Guards the trap in FINDINGS.md: under fixed density rather than fixed
    degree, the diagonal grows with p and true partial correlations vanish."""
    truth = generate(n_contexts=8, samples_per_context=2, p=p, degree=degree,
                     context_dim=3, seed=0)
    _, partial = cov_to_partial(truth["covs"])
    iu = np.triu_indices(p, k=1)
    on_edge = truth["support"][iu[0], iu[1]]
    assert np.abs(partial[:, iu[0], iu[1]][:, on_edge]).mean() > 0.05
