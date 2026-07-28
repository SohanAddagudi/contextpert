"""Does the pairwise objective recover a known context-varying Gaussian?

Three sweeps, all at p=8 with a fixed-degree sparse support:

  conditions   one replicate per condition, scaling the number of conditions.
               The regime a perturbation atlas occupies, and the one where a
               per-condition covariance does not exist at all.
  allocation   total observations held fixed, varying how they are split between
               conditions and replicates. Shows where per-condition estimation
               becomes competitive.
  replicates   fixed 200 conditions, scaling replicates. Reported for
               completeness; 200 conditions is too little total information to
               separate the estimators cleanly.

Usage:
    python -m studies.recovery --sweep conditions
"""

import argparse
import time

import numpy as np
import torch

from simgauss.estimators import (ExactGaussian, Pseudolikelihood, grouped_mle,
                                 population_mle, train_contextual)
from simgauss.generators import generate
from simgauss.metrics import evaluate
from studies._io import RESULTS, write_rows

SWEEPS = {
    "conditions": [(n, 1) for n in (100, 400, 1600, 6400)],
    "allocation": [(128, 25), (320, 10), (640, 5), (1600, 2), (3200, 1)],
    "replicates": [(200, m) for m in (2, 5, 10, 25, 50)],
}


def run_setting(p, degree, n_contexts, m, seed, context_dim, epochs):
    truth = generate(n_contexts, m, p, degree, context_dim, seed)
    x_flat = truth["x_train"].reshape(-1, p)
    context_rows = np.repeat(truth["contexts"], m, axis=0)

    fits = {"population": population_mle(x_flat, n_contexts),
            "grouped": grouped_mle(truth["x_train"])}
    for name, cls in [("pseudolikelihood", Pseudolikelihood), ("exact", ExactGaussian)]:
        torch.manual_seed(seed)
        fits[name] = train_contextual(cls(context_dim, p), truth["contexts"],
                                      x_flat, context_rows, epochs=epochs, seed=seed)
    return {name: evaluate(mu, cov, truth) for name, (mu, cov) in fits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=list(SWEEPS), default="conditions")
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--context-dim", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or RESULTS / f"recovery_{args.sweep}.csv"
    degree = min(args.degree, args.p - 1)
    for n_contexts, m in SWEEPS[args.sweep]:
        for seed in args.seeds:
            t0 = time.time()
            res = run_setting(args.p, degree, n_contexts, m, seed,
                              args.context_dim, args.epochs)
            rows = [{"study": f"recovery_{args.sweep}", "estimator": est,
                     "setting": f"{n_contexts}x{m}", "n_contexts": n_contexts,
                     "samples_per_context": m, "p": args.p, "seed": seed,
                     "metric": k, "value": v}
                    for est, mets in res.items() for k, v in mets.items()]
            write_rows(out, rows)
            print(f"{n_contexts} conditions x {m} repl, seed {seed} "
                  f"({time.time() - t0:.0f}s): "
                  + "  ".join(f"{e}={m_['corr_relerr']:.3f}" for e, m_ in res.items()),
                  flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
