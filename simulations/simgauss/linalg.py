"""Matrix utilities shared by the studies.

Batched over a leading sample axis and chunked so peak memory stays bounded.
Work is done in float64 because these feed parameter-recovery metrics, where
float32 error is comparable to the effect sizes being measured.

The projection uses a symmetric eigendecomposition and reconstructs by transpose.
A general eigendecomposition with an explicit inverse (as in some earlier code)
can return complex eigenvalues for a symmetric input and needs two passes to be
reliable; this needs one.
"""

import numpy as np

CHUNK = 4096


def _chunks(n, size=CHUNK):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def symmetrize(A):
    return (A + np.swapaxes(A, -1, -2)) / 2.0


def min_eigenvalue(A):
    """Smallest eigenvalue per matrix, for validity auditing."""
    A = np.asarray(A, dtype=np.float64)
    out = np.empty(A.shape[0], dtype=np.float64)
    for sl in _chunks(A.shape[0]):
        out[sl] = np.linalg.eigvalsh(symmetrize(A[sl]))[:, 0]
    return out


def is_pd(A, tol=0.0):
    return min_eigenvalue(A) > tol


def to_pd(A, floor=1e-6):
    """Project onto the PSD cone by clipping eigenvalues.

    Returns the projected batch and the per-matrix Frobenius distance moved,
    which is the quantity the projection-divergence tables report.
    """
    A = np.asarray(A, dtype=np.float64)
    out = np.empty_like(A)
    dist = np.empty(A.shape[0], dtype=np.float64)
    for sl in _chunks(A.shape[0]):
        sym = symmetrize(A[sl])
        vals, vecs = np.linalg.eigh(sym)
        clipped = np.clip(vals, floor, None)
        proj = (vecs * clipped[:, None, :]) @ np.swapaxes(vecs, -1, -2)
        out[sl] = proj
        dist[sl] = np.linalg.norm(proj - sym, axis=(-2, -1))
    return out, dist


def cov_to_corr(cov):
    cov = np.asarray(cov, dtype=np.float64)
    sd = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    return cov / (sd[..., :, None] * sd[..., None, :])


def cov_to_corr_safe(cov):
    """Correlation from a possibly-indefinite covariance; clips negative variances."""
    var = np.diagonal(cov, axis1=-2, axis2=-1)
    sd = np.sqrt(np.clip(var, 1e-12, None))
    return cov / (sd[..., :, None] * sd[..., None, :])


def cov_to_partial(cov):
    """Precision and partial correlations from a covariance.

    Partial correlations are scale free, so this is also the route from a
    correlation matrix to the scaled precision with no marginal variances.
    """
    prec = np.linalg.inv(cov)
    d = np.sqrt(np.diagonal(prec, axis1=-2, axis2=-1))
    partial = -prec / (d[..., :, None] * d[..., None, :])
    idx = np.arange(partial.shape[-1])
    partial[..., idx, idx] = 1.0
    return prec, partial


def upper_triangle(A, k=1):
    A = np.asarray(A)
    iu = np.triu_indices(A.shape[-1], k=k)
    return A[..., iu[0], iu[1]]
