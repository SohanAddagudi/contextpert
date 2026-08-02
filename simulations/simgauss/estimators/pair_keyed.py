"""Encoder keyed by node embeddings rather than node position.

This is what makes the two structural properties possible. Cost per step scales
with the number of pairs actually sampled, because the head is only evaluated on
the pairs in the batch; and a trained model can be evaluated on nodes absent
from training, because node identity is an input rather than a parameter index.

A dense head emitting all p^2 entries can do neither, which is why the recovery
study's estimator is not the right vehicle for the scale experiments.
"""

import numpy as np
import torch
import torch.nn as nn


class PairEncoder(nn.Module):
    """Maps (context, embedding_i, embedding_j) to a pairwise coefficient.

    Evaluating k pairs costs O(k) rather than O(p^2). The optional scale head
    supplies per-node log standard deviations for the marginal term; the
    large-scale study omits it and scores correlations directly.
    """

    def __init__(self, context_dim, embed_dim, width=96, with_scale=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + 2 * embed_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1))
        self.scale = nn.Sequential(
            nn.Linear(context_dim + embed_dim, width), nn.ReLU(),
            nn.Linear(width, 1)) if with_scale else None

    def forward(self, c, ei, ej):
        n, k = c.shape[0], ei.shape[0]
        return self.net(torch.cat([
            c.unsqueeze(1).expand(n, k, c.shape[1]),
            ei.unsqueeze(0).expand(n, k, -1),
            ej.unsqueeze(0).expand(n, k, -1)], dim=2)).squeeze(-1)

    def log_sd(self, c, e):
        n, k = c.shape[0], e.shape[0]
        return self.scale(torch.cat([
            c.unsqueeze(1).expand(n, k, c.shape[1]),
            e.unsqueeze(0).expand(n, k, -1)], dim=2)).squeeze(-1).clamp(-5, 5)


def pair_loss(model, c, x, emb, pair_i, pair_j):
    """Composite loss restricted to the given pairs. Cost scales with len(pair_i)."""
    beta = model(c, emb[pair_i], emb[pair_j])
    pairwise = ((x[:, pair_j] - x[:, pair_i] * beta) ** 2).mean()
    if model.scale is None:
        return pairwise
    nodes = torch.unique(torch.cat([pair_i, pair_j]))
    ls = model.log_sd(c, emb[nodes])
    marginal = 0.5 * (2 * ls + x[:, nodes] ** 2 / torch.exp(2 * ls)).mean()
    return pairwise + marginal


@torch.no_grad()
def predict_corr(model, c, emb, pi, pj):
    """Predicted correlations for the given pairs, assembled as in the dense case."""
    b_ij = model(c, emb[pi], emb[pj])
    b_ji = model(c, emb[pj], emb[pi])
    return (torch.sign(b_ij) * torch.sqrt(torch.clamp(b_ij * b_ji, min=0))).numpy()
