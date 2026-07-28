# Simulation studies

Synthetic studies for the contextual Gaussian graphical model estimator: does the
pairwise objective recover a known network, and what does the marginal
formulation buy computationally.

Self-contained. Nothing here imports `contextpert`, and the estimators are
written by hand rather than reusing the `Contextualized` library, so the
objective under comparison is visible in one file and the comparison is
auditable. The tradeoff is that these are reimplementations of the shipped
estimator rather than the shipped estimator itself.

Real-data work on LINCS lives outside this folder; see
`example_submissions/sm_cohesion_precision_submission.py` and
`drug_target_precision_submission.py`.

## Layout

```
simgauss/                  library
  linalg.py                PSD projection, correlation <-> partial correlation
  metrics.py               recovery metrics, edge AUROC, projection distance
  budget.py                memory sizing, so a sweep cannot take the machine down
  generators/
    sparse_precision.py    fixed-degree precision-first, known edge support
    latent_factor.py       dense rank-r, scales to p = 20,000
  estimators/
    contextual.py          pseudolikelihood and exact NLL, shared encoder
    baselines.py           population and per-condition MLE
    pair_keyed.py          encoder keyed by node embedding rather than position
studies/                   one runner per question
tests/                     analytic checks
results/                   committed CSVs, so tables regenerate without rerunning
```

## The four questions

`studies/recovery.py` asks whether the pairwise objective recovers a known
context-varying Gaussian, against a population model, a per-condition empirical
estimator, and an exact multivariate NLL. Three sweeps: `conditions` scales the
number of conditions at one replicate each, which is the regime a perturbation
atlas occupies and where a per-condition covariance does not exist at all;
`allocation` holds the total observation budget fixed and varies how it splits
between conditions and replicates, showing where per-condition estimation
becomes competitive; `replicates` fixes 200 conditions and scales replication,
kept for completeness.

`studies/cost.py` measures per-step cost against two implementations of the
exact likelihood. `studies/subsample_cost.py` measures how per-step cost falls
when only a fraction of pairs is sampled.

`studies/memory.py` finds the dimension at which the exact likelihood cannot
allocate a single gradient step, at batch size 1, which is the most favourable
possible setting for it.

`studies/scale.py` trains at p = 20,000 against known ground truth and tests
transfer to nodes held out of every training pair.

## Running

```
make test        # analytic checks, seconds
make all         # everything, roughly 70 minutes on 4 CPU cores
make recovery    # or a single study
```

Individual sweeps take arguments:

```
python -m studies.recovery --sweep allocation --seeds 0 1 2
python -m studies.scale --p 20000 --time-budget 300
```

Results append to `results/` as tidy rows, one per
(study, estimator, setting, seed, metric, value). Writes take an exclusive lock,
so sweeps can safely run as concurrent processes.

## Reading the metrics

Recovery is reported as relative Frobenius error, where 1.0 is the error of
predicting all zeros; anything above 1.0 is worse than predicting no network.
Edge recovery is AUROC over the true support, scored by absolute partial
correlation. Both are reported before and after PSD projection, because
assembled pairwise estimates are expected to be indefinite when data are scarce
and the question is how far projection moves them, not whether they are valid.

The large-scale study uses the dense latent-factor generator, which has no
sparse support, so edge AUROC is undefined there and it reports Pearson
correlation between predicted and true pairwise correlations instead. These are
different metrics and should not be compared across studies.

## Hardware

All committed results are from 4 CPU cores, 15 GB RAM, no GPU. The memory study
is the only one whose numbers depend on available RAM; it reports the machine
alongside the failure point rather than quoting a universal threshold. The cost
study's parallelism gap would be larger on a GPU, so its CPU numbers are
conservative.

Read `FINDINGS.md` before changing a generator. Three of the design choices here
look arbitrary and are not.
