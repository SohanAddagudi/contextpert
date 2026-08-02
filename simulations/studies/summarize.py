"""Regenerate the paper tables from committed results.

Results are stored as tidy rows so a study can add a metric without
invalidating existing files; this pivots them into the tables that appear in the
paper and the reviewer responses.

Usage:
    python -m studies.summarize                    # all tables
    python -m studies.summarize --study recovery_conditions
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
ORDER = ["population", "grouped", "pseudolikelihood", "exact"]
LABEL = {"population": "Population", "grouped": "Per-condition",
         "pseudolikelihood": "Ours", "exact": "Exact NLL"}
METRICS = [
    ("edge_auroc", "Edge recovery, AUROC on the true support"),
    ("corr_relerr", "Correlation recovery, relative Frobenius error"),
    ("partial_relerr", "Partial correlation recovery, relative Frobenius error"),
    ("corr_relerr_preproj", "Correlation recovery before PSD projection"),
    ("nonpd_frac", "Fraction of assembled matrices indefinite before projection"),
]
INDEX = {"recovery_conditions": "n_contexts",
         "recovery_allocation": "setting",
         "recovery_replicates": "samples_per_context",
         "recovery_dimension_grid": ["p", "samples_per_context"]}


def markdown(df, index_name):
    head = "| " + " | ".join([index_name] + [str(c) for c in df.columns]) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows = ["| " + " | ".join([str(i)] + [f"{v:.3f}" for v in r]) + " |"
            for i, r in df.iterrows()]
    return "\n".join([head, sep] + rows)


def recovery_tables(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    idx = INDEX[name]
    print(f"\n## {name}\n")
    for metric, title in METRICS:
        sub = df[df.metric == metric]
        if sub.empty:
            continue
        t = sub.pivot_table(index=idx, columns="estimator", values="value", aggfunc="mean")
        t = t[[c for c in ORDER if c in t.columns]].rename(columns=LABEL)
        print(f"**{title}**\n")
        print(markdown(t, idx if isinstance(idx, str) else "setting") + "\n")


def cost_table():
    path = RESULTS / "cost.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    d["us_per_sample"] = d.ms_loss_only / d.batch * 1000
    t = d.pivot_table(index="p", columns="variant", values="us_per_sample")
    t = t[[c for c in ["pairwise", "exact_factor", "exact_cov"] if c in t.columns]]
    t.columns = ["Ours", "Exact NLL (factor)", "Exact NLL (covariance)"][:len(t.columns)]
    print("\n## cost: loss cost, microseconds per sample\n")
    print(markdown(t, "Dimension") + "\n")


def memory_table():
    path = RESULTS / "memory.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    d["cell"] = d.apply(
        lambda r: "out of memory" if r.status != "ok" else f"{r.peak_rss_mb:.0f} MB", axis=1)
    t = d.pivot_table(index="p", columns="estimator", values="cell", aggfunc="first")
    print("\n## memory: peak resident memory for one gradient step, batch size 1\n")
    head = "| Dimension | " + " | ".join(t.columns) + " |"
    print("\n".join([head, "| --- " * (len(t.columns) + 1) + "|"]
                    + ["| " + " | ".join([str(i)] + list(r)) + " |" for i, r in t.iterrows()]) + "\n")


def scale_table():
    path = RESULTS / "scale.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    f = d.sort_values("seconds").groupby("budget").tail(1).set_index("budget")
    t = f[["step", "seconds", "seen_pearson", "held_pearson"]]
    t.columns = ["Steps", "Seconds", "Seen nodes (r)", "Held-out nodes (r)"]
    print("\n## scale: final state at equal wall-clock\n")
    print(markdown(t, "Pairs per step") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=None)
    args = ap.parse_args()
    names = [args.study] if args.study else list(INDEX)
    for n in names:
        if n in INDEX:
            recovery_tables(n)
    if not args.study:
        cost_table()
        memory_table()
        scale_table()


if __name__ == "__main__":
    main()
