"""X2: cost of the pairwise surrogate against the exact multivariate NLL.

No training and no ground truth; this measures the cost of a step, not accuracy.
Three variants are compared because the honest answer depends on how the exact
likelihood is parameterized:

  pairwise      our composite surrogate, elementwise over p^2 pairs
  exact_factor  head predicts a triangular factor; log determinant is free off
                the diagonal and the quadratic form is a triangular solve, so
                this is also quadratic per sample and is the fair comparison
  exact_cov     head predicts a covariance, which must be factorized per sample
                per step, making it cubic per sample

Reporting only exact_cov would overstate our case. Reporting both bounds it.

Each configuration runs in a fresh subprocess so that peak resident memory is
attributable to that configuration rather than to everything run before it.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simgauss.budget import check, safe_batch  # noqa: E402

LOG_2PI = 1.8378770664093453


def build(variant, p, batch, context_dim=8, width=64):
    head_out = {
        "pairwise": p + p + p * p,
        "exact_factor": p + p * (p + 1) // 2,
        "exact_cov": p + p * p,
    }[variant]
    head = torch.nn.Sequential(
        torch.nn.Linear(context_dim, width), torch.nn.ReLU(),
        torch.nn.Linear(width, head_out),
    )
    c = torch.randn(batch, context_dim)
    x = torch.randn(batch, p)
    return head, c, x


def step(variant, head, c, x, p, with_loss=True):
    out = head(c)
    if not with_loss:
        return out.sum()
    if variant == "pairwise":
        mu = out[:, :p]
        log_sd = out[:, p:2 * p].clamp(-5, 5)
        beta = out[:, 2 * p:].reshape(-1, p, p)
        resid = x - mu
        pred = resid.unsqueeze(2) * beta
        target = resid.unsqueeze(1).expand_as(pred)
        pairwise = ((target - pred) ** 2).mean(dim=(1, 2))
        var = torch.exp(2 * log_sd)
        marginal = 0.5 * (LOG_2PI + 2 * log_sd + resid ** 2 / var).sum(dim=1)
        return (pairwise + marginal / p).mean()
    if variant == "exact_factor":
        mu = out[:, :p]
        tril = torch.tril_indices(p, p)
        L = torch.zeros(c.shape[0], p, p)
        L[:, tril[0], tril[1]] = out[:, p:]
        diag = torch.diagonal(L, dim1=1, dim2=2)
        L = L - torch.diag_embed(diag) + torch.diag_embed(torch.nn.functional.softplus(diag) + 1e-4)
        z = torch.linalg.solve_triangular(L, (x - mu).unsqueeze(2), upper=False).squeeze(2)
        logdet = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        return (0.5 * p * LOG_2PI + logdet + 0.5 * (z ** 2).sum(1)).mean()
    if variant == "exact_cov":
        mu = out[:, :p]
        A = out[:, p:].reshape(-1, p, p)
        # symmetrize and shift onto the cone, then factorize: the cubic path
        S = A @ A.transpose(1, 2) + p * torch.eye(p).unsqueeze(0)
        L = torch.linalg.cholesky(S)
        z = torch.linalg.solve_triangular(L, (x - mu).unsqueeze(2), upper=False).squeeze(2)
        logdet = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        return (0.5 * p * LOG_2PI + logdet + 0.5 * (z ** 2).sum(1)).mean()
    raise ValueError(variant)


def time_variant(variant, p, batch, trials, warmup, with_loss):
    head, c, x = build(variant, p, batch)
    opt = torch.optim.SGD(head.parameters(), lr=0.0)  # zero lr: cost only, no drift
    for _ in range(warmup):
        opt.zero_grad()
        step(variant, head, c, x, p, with_loss).backward()
        opt.step()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        opt.zero_grad()
        step(variant, head, c, x, p, with_loss).backward()
        opt.step()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def run_single(args):
    torch.set_num_threads(args.threads)
    batch = args.batch or safe_batch(args.p)
    check(args.p, batch)
    full = time_variant(args.variant, args.p, batch, args.trials, args.warmup, True)
    head_only = time_variant(args.variant, args.p, batch, args.trials, args.warmup, False)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({
        "variant": args.variant, "p": args.p, "batch": batch,
        "ms_head_plus_loss": round(full, 3), "ms_head_only": round(head_only, 3),
        "ms_loss_only": round(full - head_only, 3), "peak_rss_mb": round(peak_mb, 1),
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--p", type=int)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--cov-max-p", type=int, default=1024,
                    help="exact_cov is cubic per sample; beyond this it costs minutes per trial")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "cost.csv"))
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    import csv
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in args.dims:
        for variant in ["pairwise", "exact_factor", "exact_cov"]:
            if variant == "exact_cov" and p > args.cov_max_p:
                print(f"skip {variant} p={p}: above the stated cap", flush=True)
                continue
            cmd = [sys.executable, __file__, "--single", "--variant", variant,
                   "--p", str(p), "--trials", str(args.trials), "--threads", str(args.threads)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"FAILED {variant} p={p}: {res.stderr.strip()[-300:]}", flush=True)
                continue
            row = json.loads(res.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(f"p={p:5d} {variant:13s} batch={row['batch']:4d} "
                  f"loss={row['ms_loss_only']:9.2f} ms  total={row['ms_head_plus_loss']:9.2f} ms  "
                  f"rss={row['peak_rss_mb']:7.1f} MB", flush=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
