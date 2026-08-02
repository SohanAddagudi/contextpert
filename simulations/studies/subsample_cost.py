"""X4a, cost half: does per-step cost actually track the number of sampled pairs?

The structural claim is that cost per step decouples from the square of the
dimension. That is an asymptotic statement, and asymptotic statements have a
regime where they are false. A step has fixed overhead (optimizer, Python,
the per-node scale term) that does not shrink when pairs are subsampled, so at
small dimension the overhead dominates and subsampling buys little.

This measures where the crossover is, with no training involved, so the answer
is not confounded by convergence behaviour. Reported as speedup relative to
using all pairs at the same dimension.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simgauss.estimators import PairEncoder, pair_loss  # noqa: E402


def time_step(p, fraction, args):
    torch.manual_seed(0)
    c = torch.randn(args.sample_batch, args.context_dim)
    x = torch.randn(args.sample_batch, p)
    emb = torch.randn(p, args.embed_dim)
    nodes = torch.arange(p)
    ii, jj = torch.meshgrid(nodes, nodes, indexing="ij")
    pi_all, pj_all = ii.reshape(-1), jj.reshape(-1)
    n_pairs = len(pi_all)
    k = max(1, int(round(fraction * n_pairs)))

    model = PairEncoder(args.context_dim, args.embed_dim, width=args.width, with_scale=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def one():
        sel = torch.randint(0, n_pairs, (k,)) if k < n_pairs else None
        pi, pj = (pi_all[sel], pj_all[sel]) if sel is not None else (pi_all, pj_all)
        opt.zero_grad()
        pair_loss(model, c, x, emb, pi, pj).backward()
        opt.step()

    for _ in range(args.warmup):
        one()
    ts = []
    for _ in range(args.trials):
        t0 = time.perf_counter()
        one()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2], k, n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1, 0.05])
    ap.add_argument("--sample-batch", type=int, default=64)
    ap.add_argument("--context-dim", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=6)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "subsample_cost.csv"))
    args = ap.parse_args()

    torch.set_num_threads(4)
    rows = []
    for p in args.dims:
        base = None
        for frac in args.fractions:
            ms, k, n_pairs = time_step(p, frac, args)
            if frac == 1.0:
                base = ms
            speedup = base / ms if base else float("nan")
            rows.append({"p": p, "fraction": frac, "pairs_used": k, "pairs_total": n_pairs,
                         "ms_per_step": round(ms, 3), "speedup_vs_full": round(speedup, 2)})
            print(f"p={p:4d} frac={frac:<5} pairs={k:6d}/{n_pairs:<6d} "
                  f"{ms:8.2f} ms  speedup={speedup:5.2f}x", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
