"""Recovery metrics against known ground truth.

Extracted so every study scores the same way. Previously these lived inside the
recovery runner, which is why the large-scale study had to reimplement them
slightly differently and the two were not directly comparable.

Reporting is split before and after PSD projection. Assembled pairwise estimates
are expected to be indefinite when data are scarce, so the question is not
whether they are valid but how far projection moves them and whether the
projected estimate stays faithful.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .linalg import (cov_to_corr, cov_to_corr_safe, cov_to_partial,
                     min_eigenvalue, to_pd, upper_triangle)


def rel_frob(est, true):
    """Relative Frobenius error. 1.0 is the error of predicting all zeros."""
    return float(np.linalg.norm(est - true) / np.linalg.norm(true))


def mean_loglik(x_eval, mu, cov):
    """Mean log-likelihood of held-out samples. NaN if any covariance is singular."""
    n_contexts, k, p = x_eval.shape
    total = 0.0
    for c in range(n_contexts):
        sign, logdet = np.linalg.slogdet(cov[c])
        if sign <= 0:
            return float("nan")
        inv = np.linalg.inv(cov[c])
        d = x_eval[c] - mu[c]
        total += np.sum(-0.5 * (p * np.log(2 * np.pi) + logdet
                                + np.einsum("ij,jk,ik->i", d, inv, d)))
    return float(total / (n_contexts * k))


def evaluate(mu_hat, cov_hat, truth):
    """All recovery metrics for one estimator against one ground-truth draw."""
    out = {}
    true_cov = truth["covs"]
    true_corr = cov_to_corr(true_cov)
    _, true_partial = cov_to_partial(true_cov)
    support = upper_triangle(truth["support"][None, ...])[0].astype(int)

    eigs = min_eigenvalue(cov_hat)
    out["nonpd_frac"] = float((eigs <= 0).mean())
    out["corr_relerr_preproj"] = rel_frob(upper_triangle(cov_to_corr_safe(cov_hat)),
                                          upper_triangle(true_corr))

    cov_pd, dist = to_pd(cov_hat)
    norms = np.linalg.norm(cov_hat, axis=(1, 2))
    # A degenerate estimate (zero covariance, e.g. one replicate per condition)
    # has no scale to normalize by; report 0 rather than dividing by zero.
    out["proj_reldist"] = float(np.median(
        np.where(norms > 0, dist / np.where(norms > 0, norms, 1), 0.0)))

    est_corr = cov_to_corr(cov_pd)
    _, est_partial = cov_to_partial(cov_pd)
    out["cov_relerr"] = rel_frob(cov_pd, true_cov)
    out["corr_relerr"] = rel_frob(upper_triangle(est_corr), upper_triangle(true_corr))
    out["partial_relerr"] = rel_frob(upper_triangle(est_partial),
                                     upper_triangle(true_partial))

    # Edge score is |partial correlation|, averaged over conditions. Undefined
    # when the support is all-ones or all-zeros, e.g. at p=2 with one edge.
    scores = np.abs(upper_triangle(est_partial)).mean(axis=0)
    if 0 < support.sum() < len(support):
        out["edge_auroc"] = float(roc_auc_score(support, scores))
        out["edge_auprc"] = float(average_precision_score(support, scores))
    out["mll"] = mean_loglik(truth["x_eval"], mu_hat, cov_pd)
    return out


def pearson(est, true):
    """Correlation between predicted and true values, for the dense large-p study
    where there is no sparse support and edge AUROC is undefined."""
    return float(np.corrcoef(np.asarray(est).ravel(), np.asarray(true).ravel())[0, 1])
