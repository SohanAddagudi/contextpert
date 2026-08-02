"""Precision-first generator with a known sparse support.

We build the precision matrix rather than the covariance. Sampling a covariance
as a random matrix times its transpose produces mostly small off-diagonal values
once the dimension grows, which makes recovery uninteresting.

Building the precision directly fixes three things at once: magnitudes are set
explicitly so edges stay detectable, strict diagonal dominance guarantees
positive definiteness at every context and dimension, and the sparsity pattern
is known, so edge recovery is measurable against a real support.

Degree is held constant as p grows, not density. Under fixed density each row
accumulates more off-diagonal mass, diagonal dominance pushes the diagonal up,
and the induced partial correlations shrink back toward zero. See FINDINGS.md.
"""

import numpy as np


def band_mask(p, degree):
    """Fixed-degree pattern: each node connects to its next `degree` neighbours."""
    return [(i, i + k) for i in range(p) for k in range(1, degree + 1) if i + k < p]


def make_precisions(contexts, p, degree, rng, amp_low=0.3, amp_high=0.8, ridge=0.1):
    n, q = contexts.shape
    edges = band_mask(p, degree)
    amps = rng.uniform(amp_low, amp_high, len(edges)) * rng.choice([-1.0, 1.0], len(edges))
    directions = rng.normal(size=(len(edges), q))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    thetas = np.zeros((n, p, p))
    activations = np.tanh(contexts @ directions.T)
    for e, (i, j) in enumerate(edges):
        vals = amps[e] * activations[:, e]
        thetas[:, i, j] = vals
        thetas[:, j, i] = vals

    idx = np.arange(p)
    thetas[:, idx, idx] = ridge + np.abs(thetas).sum(axis=2)
    return thetas, edges


def generate(n_contexts, samples_per_context, p, degree, context_dim, seed,
             samples_held_out=20, mean_scale=1.0):
    """Draw contexts, build true Gaussians, sample observations.

    Returns training observations, held-out observations from the same contexts
    for likelihood evaluation, and the ground-truth parameters. Recovery is
    scored on the training contexts: this measures estimation, not extrapolation
    to unseen contexts, which is a separate question we do not claim.
    """
    rng = np.random.default_rng(seed)
    contexts = rng.uniform(-1, 1, (n_contexts, context_dim))
    thetas, edges = make_precisions(contexts, p, degree, rng)
    covs = np.linalg.inv(thetas)

    mus = contexts @ rng.normal(scale=mean_scale, size=(context_dim, p))

    def draw(k):
        out = np.zeros((n_contexts, k, p))
        for c in range(n_contexts):
            out[c] = rng.multivariate_normal(mus[c], covs[c], size=k)
        return out

    support = np.zeros((p, p), dtype=bool)
    for i, j in edges:
        support[i, j] = support[j, i] = True

    return {"contexts": contexts, "x_train": draw(samples_per_context),
            "x_eval": draw(samples_held_out), "mus": mus, "covs": covs,
            "precisions": thetas, "support": support, "edges": edges}
