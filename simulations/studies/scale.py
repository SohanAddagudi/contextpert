"""Training at transcriptome scale, and transfer to nodes absent from training.

The memory study establishes that an exact likelihood cannot allocate a single
gradient step beyond roughly p=4096 on commodity hardware. That only matters if
the pair-keyed estimator does something useful at those dimensions, which is
what this measures, at p=20,000 and 4x10^8 ordered pairs.

Two questions:

1. Does a pair budget converge, and is a smaller budget better per second? Arms
   are compared at equal wall-clock, which is the fair comparison; equal step
   counts would simply reward the arm doing more work per step.

2. Do predictions transfer to nodes excluded from every training pair? An exact
   likelihood cannot be evaluated in this setting at all, because its parameter
   vector is tied to the training feature set.

Ground truth here is the dense latent-factor generator, so there is no sparse
support and edge AUROC is undefined; we score Pearson correlation between
predicted and true pairwise correlations.

Usage:
    python -m studies.scale --p 20000 --time-budget 300
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simgauss.estimators import PairEncoder, pair_loss, predict_corr  # noqa: E402
from simgauss.generators import LatentFactor  # noqa: E402
from simgauss.metrics import pearson, rel_frob  # noqa: E402
from studies._io import RESULTS, write_rows  # noqa: E402

FIELDS = ["study", "budget", "step", "seconds", "seen_pearson", "held_pearson",
          "seen_relerr", "held_relerr"]


def eval_pairs(rng, pool, n, gen, U):
    """A balanced evaluation pair set drawn from `pool`, with its true correlations.

    Evaluating on sampled pairs keeps evaluation O(k) rather than O(p^2), for
    the same reason the training objective is tractable at this scale.
    """
    pi = rng.choice(pool, size=n)
    pj = rng.choice(pool, size=n)
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]
    return torch.tensor(pi), torch.tensor(pj), gen.true_corr(U, pi, pj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=20000)
    ap.add_argument("--rank", type=int, default=5)
    ap.add_argument("--contexts", type=int, default=60)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--context-dim", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=12)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sample-batch", type=int, default=64)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 4000, 16000])
    ap.add_argument("--time-budget", type=float, default=300.0,
                    help="wall-clock seconds per arm; equal time is the fair comparison")
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or RESULTS / "scale.csv"
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)

    gen = LatentFactor(args.p, args.embed_dim, args.context_dim, args.rank, args.seed)
    contexts = rng.uniform(-1, 1, (args.contexts, args.context_dim))
    x, U = gen.draw(contexts, args.samples, rng)
    print(f"p={args.p:,}  ordered pairs={args.p ** 2:,}  "
          f"samples={args.contexts * args.samples}", flush=True)

    n_hold = int(args.p * args.holdout_frac)
    held = np.sort(rng.choice(args.p, size=n_hold, replace=False))
    seen = np.setdiff1d(np.arange(args.p), held)
    print(f"training nodes={len(seen):,}  held-out nodes={len(held):,}", flush=True)

    n_eval_ctx = min(20, args.contexts)
    c_eval = torch.tensor(contexts[:n_eval_ctx], dtype=torch.float32)
    emb_t = torch.tensor(gen.emb, dtype=torch.float32)
    ev = {name: eval_pairs(rng, pool, 6000, gen, U[:n_eval_ctx])
          for name, pool in [("seen", seen), ("held", held)]}
    for name, (pi, _, tc) in ev.items():
        print(f"  {name}: {len(pi)} eval pairs, true corr std {tc.std():.3f}", flush=True)

    c_all = torch.tensor(np.repeat(contexts, args.samples, axis=0), dtype=torch.float32)
    x_all = torch.tensor(x.reshape(-1, args.p), dtype=torch.float32)
    del x
    seen_t = torch.tensor(seen)

    for k in args.budgets:
        torch.manual_seed(args.seed)
        model = PairEncoder(args.context_dim, args.embed_dim, args.width)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        elapsed, rows = 0.0, []
        for step in range(1, 10_000_000):
            t0 = time.perf_counter()
            # Pairs are drawn only from training nodes, so held-out nodes never
            # appear in any gradient.
            pi = seen_t[torch.randint(0, len(seen_t), (k,))]
            pj = seen_t[torch.randint(0, len(seen_t), (k,))]
            idx = torch.randint(0, c_all.shape[0], (args.sample_batch,))
            opt.zero_grad()
            pair_loss(model, c_all[idx], x_all[idx], emb_t, pi, pj).backward()
            opt.step()
            elapsed += time.perf_counter() - t0
            done = elapsed >= args.time_budget
            if step % args.eval_every == 0 or done:
                scores = {}
                for name, (epi, epj, tc) in ev.items():
                    pred = predict_corr(model, c_eval, emb_t, epi, epj)
                    scores[name] = (pearson(pred, tc), rel_frob(pred, tc))
                rows.append({"study": "scale", "budget": k, "step": step,
                             "seconds": round(elapsed, 2),
                             "seen_pearson": round(scores["seen"][0], 4),
                             "held_pearson": round(scores["held"][0], 4),
                             "seen_relerr": round(scores["seen"][1], 4),
                             "held_relerr": round(scores["held"][1], 4)})
                print(f"k={k:6d} step={step:6d} {elapsed:6.1f}s  "
                      f"seen r={scores['seen'][0]:.3f} | held r={scores['held'][0]:.3f}",
                      flush=True)
            if done:
                break
        write_rows(out, rows, fields=FIELDS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
