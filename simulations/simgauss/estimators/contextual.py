"""The two contextual estimators being compared.

Both share the encoder architecture, optimizer and epoch budget, so the only
difference between them is the objective. That is the point of the comparison:
any gap in recovery is attributable to the loss, not to capacity or tuning.

Pseudolikelihood assembles a covariance from independently estimated pairwise
regressions, which carries no positive-definiteness guarantee and is repaired by
projection afterwards. ExactGaussian parameterizes a triangular factor, so its
covariance is valid by construction and never needs projecting. Comparing the
two after projection measures what the repair costs.
"""

import numpy as np
import torch
import torch.nn as nn

LOG_2PI = float(np.log(2 * np.pi))


def mlp(in_dim, out_dim, width=64, depth=2):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, out_dim))


class Pseudolikelihood(nn.Module):
    """Composite likelihood: pairwise regressions plus per-feature Gaussian terms.

    The two terms are weighted so each is a per-element mean: the pairwise term
    averages over p^2 ordered pairs and the marginal term over p features. This
    keeps them on a comparable scale as p varies. The weighting is a choice, not
    derived; results are not sensitive to it in the range we tested.
    """

    def __init__(self, context_dim, p, width=64):
        super().__init__()
        self.p = p
        self.encoder = mlp(context_dim, p + p + p * p, width)

    def forward(self, c):
        out = self.encoder(c)
        p = self.p
        return out[:, :p], out[:, p:2 * p].clamp(-5.0, 5.0), out[:, 2 * p:].reshape(-1, p, p)

    def loss(self, c, x):
        mu, log_sd, beta = self(c)
        resid = x - mu
        pred = resid.unsqueeze(2) * beta                 # (n, i, j): resid_i * beta_ij
        target = resid.unsqueeze(1).expand_as(pred)      # (n, i, j): resid_j
        pairwise = ((target - pred) ** 2).mean(dim=(1, 2))
        var = torch.exp(2 * log_sd)
        marginal = 0.5 * (LOG_2PI + 2 * log_sd + resid ** 2 / var).sum(dim=1)
        return (pairwise + marginal / self.p).mean()

    @torch.no_grad()
    def covariance(self, c):
        """Assemble a covariance from pairwise coefficients and marginal scales.

        corr_ij = sign * sqrt(beta_ij * beta_ji). Pairs whose two coefficients
        disagree in sign carry no consistent dependency and are set to zero.
        """
        mu, log_sd, beta = self(c)
        prod = beta * beta.transpose(1, 2)
        sign = torch.sign(beta)
        consistent = sign == sign.transpose(1, 2)
        corr = torch.where(consistent, sign * torch.sqrt(prod.clamp(min=0)),
                           torch.zeros_like(prod))
        sd = torch.exp(log_sd)
        return mu.numpy(), (corr * (sd.unsqueeze(2) * sd.unsqueeze(1))).numpy()


class ExactGaussian(nn.Module):
    """Exact multivariate NLL with a triangular-factor head, valid by construction."""

    def __init__(self, context_dim, p, width=64):
        super().__init__()
        self.p = p
        self.tril_idx = torch.tril_indices(p, p)
        self.encoder = mlp(context_dim, p + self.tril_idx.shape[1], width)

    def factor(self, c):
        out = self.encoder(c)
        p = self.p
        mu, raw = out[:, :p], out[:, p:]
        L = torch.zeros(c.shape[0], p, p, dtype=out.dtype, device=out.device)
        L[:, self.tril_idx[0], self.tril_idx[1]] = raw
        diag = torch.diagonal(L, dim1=1, dim2=2)
        # softplus keeps the diagonal positive, so L L^T is positive definite
        new_diag = torch.nn.functional.softplus(diag) + 1e-4
        return mu, L - torch.diag_embed(diag) + torch.diag_embed(new_diag)

    def loss(self, c, x):
        mu, L = self.factor(c)
        z = torch.linalg.solve_triangular(L, (x - mu).unsqueeze(2), upper=False).squeeze(2)
        log_det = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
        return (0.5 * self.p * LOG_2PI + log_det + 0.5 * (z ** 2).sum(dim=1)).mean()

    @torch.no_grad()
    def covariance(self, c):
        mu, L = self.factor(c)
        return mu.numpy(), (L @ L.transpose(1, 2)).numpy()


def train_contextual(model, contexts, x_flat, context_rows, epochs=600, lr=1e-3, seed=0):
    """Full-batch training; the simulation sizes here are small enough to be stable."""
    torch.manual_seed(seed)
    c = torch.tensor(context_rows, dtype=torch.float32)
    x = torch.tensor(x_flat, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        model.loss(c, x).backward()
        opt.step()
    with torch.no_grad():
        return model.covariance(torch.tensor(contexts, dtype=torch.float32))
