"""X4c: at what dimension does the exact likelihood stop fitting in memory?

This is the sharpest form of the efficiency argument, and unlike wall-clock it
does not depend on implementation quality.

An exact multivariate Gaussian likelihood must materialize a per-sample $p\\times p$
object: either the covariance, or its triangular factor, or the head that emits
$p(p+1)/2$ parameters. That is $\\Omega(p^2)$ memory per sample no matter how the
solve is arranged, and autograd holds several such tensors live at once. Batch
size can be reduced to 1, but the quadratic term remains.

Our objective is a sum over node pairs and the head is keyed by node embeddings,
so a gradient step touching $k$ sampled pairs allocates $O(k)$ regardless of $p$.
The pair budget is a free parameter, so memory is flat in dimension.

The experiment sweeps $p$ with batch fixed at 1, the most favourable setting for
the exact estimator, and records peak resident memory for each. Each
configuration runs in a fresh subprocess so peak memory is attributable, and an
out-of-memory failure is recorded as a result rather than crashing the sweep.
"""

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch


def exact_step(p, batch, context_dim=8, width=64):
    """One gradient step for an exact-NLL estimator with a triangular-factor head."""
    n_tril = p * (p + 1) // 2
    head = torch.nn.Sequential(
        torch.nn.Linear(context_dim, width), torch.nn.ReLU(),
        torch.nn.Linear(width, p + n_tril),
    )
    opt = torch.optim.SGD(head.parameters(), lr=0.0)
    c = torch.randn(batch, context_dim)
    x = torch.randn(batch, p)
    tril = torch.tril_indices(p, p)
    opt.zero_grad()
    out = head(c)
    mu = out[:, :p]
    L = torch.zeros(batch, p, p)
    L[:, tril[0], tril[1]] = out[:, p:]
    diag = torch.diagonal(L, dim1=1, dim2=2)
    L = L - torch.diag_embed(diag) + torch.diag_embed(torch.nn.functional.softplus(diag) + 1e-4)
    z = torch.linalg.solve_triangular(L, (x - mu).unsqueeze(2), upper=False).squeeze(2)
    logdet = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    loss = (logdet + 0.5 * (z ** 2).sum(1)).mean()
    loss.backward()
    opt.step()


def marginal_step(p, batch, pairs, context_dim=8, embed_dim=16, width=64):
    """One gradient step for the pair-keyed marginal estimator.

    The head consumes (context, embedding_i, embedding_j) and emits one
    coefficient per sampled pair, so nothing of size p by p is ever formed.
    """
    head = torch.nn.Sequential(
        torch.nn.Linear(context_dim + 2 * embed_dim, width), torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    )
    opt = torch.optim.SGD(head.parameters(), lr=0.0)
    c = torch.randn(batch, context_dim)
    x = torch.randn(batch, p)
    emb = torch.randn(p, embed_dim)
    pi = torch.randint(0, p, (pairs,))
    pj = torch.randint(0, p, (pairs,))
    opt.zero_grad()
    feat = torch.cat([
        c.unsqueeze(1).expand(batch, pairs, context_dim),
        emb[pi].unsqueeze(0).expand(batch, pairs, embed_dim),
        emb[pj].unsqueeze(0).expand(batch, pairs, embed_dim),
    ], dim=2)
    beta = head(feat).squeeze(-1)
    loss = ((x[:, pj] - x[:, pi] * beta) ** 2).mean()
    loss.backward()
    opt.step()


def run_single(args):
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    if args.estimator == "exact":
        exact_step(args.p, args.batch)
    else:
        marginal_step(args.p, args.batch, args.pairs)
    ms = (time.perf_counter() - t0) * 1000
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"estimator": args.estimator, "p": args.p, "batch": args.batch,
                      "pairs": args.pairs if args.estimator == "marginal" else None,
                      "peak_rss_mb": round(peak, 1), "ms": round(ms, 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--estimator", default="exact")
    ap.add_argument("--p", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "memory.csv"))
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    rows = []
    for p in args.dims:
        for est in ["exact", "marginal"]:
            cmd = [sys.executable, __file__, "--single", "--estimator", est,
                   "--p", str(p), "--batch", str(args.batch), "--pairs", str(args.pairs),
                   "--threads", str(args.threads)]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                rows.append({"estimator": est, "p": p, "batch": args.batch,
                             "pairs": args.pairs if est == "marginal" else None,
                             "peak_rss_mb": None, "ms": None, "status": "timeout"})
                print(f"p={p:6d} {est:9s} TIMEOUT after {args.timeout}s", flush=True)
                continue
            if res.returncode != 0:
                tail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unknown"
                status = "oom" if ("alloc" in tail.lower() or "memory" in tail.lower()
                                   or res.returncode == -9) else f"fail:{res.returncode}"
                rows.append({"estimator": est, "p": p, "batch": args.batch,
                             "pairs": args.pairs if est == "marginal" else None,
                             "peak_rss_mb": None, "ms": None, "status": status})
                print(f"p={p:6d} {est:9s} {status.upper()}  {tail[:90]}", flush=True)
                continue
            row = json.loads(res.stdout.strip().splitlines()[-1])
            row["status"] = "ok"
            rows.append(row)
            print(f"p={p:6d} {est:9s} peak={row['peak_rss_mb']:8.1f} MB  {row['ms']:9.1f} ms",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["estimator", "p", "batch", "pairs",
                                           "peak_rss_mb", "ms", "status"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
