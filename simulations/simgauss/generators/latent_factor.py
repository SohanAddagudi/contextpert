"""Dense latent-factor generator, scalable to very large p.

Used where a sparse precision matrix cannot be built, inverted or sampled, which
on commodity hardware is roughly p beyond a few thousand.

    Sigma(c) = D + U(c) U(c)^T,   U(c) in R^{p x rank},   rank small

with each row u_i = g(e_i, c) a function of that node's embedding and the
context. Three properties matter. Every pair carries signal, so a sampled pair is
informative; block-diagonal structure fails here because uniform pair sampling
almost never lands on an edge (see FINDINGS.md). The true correlation for any
pair is available in O(rank) from the two rows, so ground truth never requires
forming a p by p matrix. And sampling is x = sqrt(D) eps + U z with z in
R^rank, which is O(p * rank) per draw.

There is no sparse support here, so edge recovery is undefined; the large-p
study scores Pearson correlation between predicted and true pairwise
correlations instead.
"""

import numpy as np


class LatentFactor:
    def __init__(self, p, embed_dim, context_dim, rank, seed,
                 diag_lo=0.3, diag_hi=1.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.p = p
        self.rank = rank
        emb = rng.normal(size=(p, embed_dim))
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        # Loadings are a bilinear function of embedding and context, so the
        # dependency between two nodes is recoverable from their embeddings.
        self.W = rng.normal(size=(rank, embed_dim, context_dim)) / np.sqrt(context_dim)
        self.d = rng.uniform(diag_lo, diag_hi, size=p)

    def loadings(self, contexts):
        """U for every context: (n_contexts, p, rank)."""
        proj = np.einsum("pe,rec->prc", self.emb, self.W)
        return np.tanh(np.einsum("prc,nc->npr", proj, contexts))

    def draw(self, contexts, m, rng=None):
        """Sample replicates without ever forming a p by p covariance."""
        rng = rng or self.rng
        U = self.loadings(contexts)
        n_ctx, p, r = U.shape
        sd = np.sqrt(self.d)
        x = np.zeros((n_ctx, m, p))
        for k in range(n_ctx):
            z = rng.normal(size=(m, r))
            x[k] = z @ U[k].T + rng.normal(size=(m, p)) * sd
        return x, U

    def true_corr(self, U, pi, pj):
        """Exact correlation for arbitrary pairs, in O(k * rank)."""
        ui, uj = U[:, pi, :], U[:, pj, :]
        cov = (ui * uj).sum(-1)
        vi = self.d[pi] + (ui ** 2).sum(-1)
        vj = self.d[pj] + (uj ** 2).sum(-1)
        cov = np.where(pi == pj, vi, cov)
        return cov / np.sqrt(vi * vj)
