# Findings

Negative results and design traps from building these studies. Each of these
cost real time and none is obvious from the code alone, so they are recorded
here rather than lost.

## Fixed degree, not fixed density

The first generator built a dense precision matrix and enforced positive
definiteness by strict diagonal dominance. That combination is self-defeating:
with every pair connected, each diagonal entry must exceed the sum of p-1
off-diagonal magnitudes, so the diagonal grows with p while the off-diagonals do
not. The induced partial correlations collapse toward zero.

At p = 20 this produced true off-diagonal correlations with standard deviation
0.040 and a maximum near 0.13. The recovery task was vacuous: predicting zero
was very nearly the correct answer, and every estimator scored around 1.0
relative error while appearing to fail.

The fix is to hold the per-node degree constant as p grows rather than the
density. Each row then carries a bounded number of off-diagonal terms, the
diagonal stays small, and per-edge partial correlations remain detectable at any
dimension. `tests/test_linalg.py::test_fixed_degree_keeps_partial_correlations_detectable`
guards this.

## Block-diagonal structure defeats uniform pair subsampling

The first attempt at the large-p study used a block-diagonal precision matrix,
which makes generation and inversion cheap at p in the thousands. It does not
work with a pair-subsampling objective.

With blocks of 20 inside p = 2000, the probability that a uniformly sampled pair
falls inside a block is about 1 percent. A step sampling 1000 pairs sees roughly
10 true edges and 990 true zeros, so the gradient is dominated by the trivial
part of the problem and training learns to predict nothing. The k = 1000 arm sat
at AUROC 0.53 after 5757 steps.

Real expression covariance is dense and driven by a small number of latent
programs, which is both more faithful and the regime where pair subsampling is
sensible. The latent-factor generator replaced it: every pair carries signal,
true correlations for any pair are available in O(rank) from two rows, and
sampling is O(p * rank) per draw, so nothing of size p by p is ever formed. That
is what makes p = 20,000 reachable at all.

## Node transfer needs enough nodes and structured embeddings

An early version of the transfer experiment held out nodes from a 20-node
problem with 14 training nodes and random, unstructured embeddings. It failed
completely: held-out error was worse than predicting no dependency, and it grew
as training proceeded while seen-node error fell. The natural reading is that
transfer does not work.

That reading was wrong. With 14 nodes and embeddings unrelated to the covariance,
there is no embedding-to-dependency mapping available to learn, so the model
memorised node identity, which is exactly what the failure signature showed.
Rerun with 18,000 training nodes and embeddings that genuinely determine the
covariance, held-out recovery tracks seen-node recovery to within 0.004 Pearson.

The lesson is about experimental power rather than the method: a transfer test
run below the scale at which transfer is learnable measures nothing.

## The cost comparison does not favour the marginal objective

Measured against an exact likelihood whose head predicts a triangular factor,
which is the fair comparison, per-step cost converges to parity: 1.25 times
faster at p = 8, 1.10 at 128, 1.01 at 1024. With a dense head over all p^2 pairs
our peak memory is also somewhat higher.

Both objectives are quadratic per sample. The gap that does exist, 1.65 to 5.84
times, is against an implementation that predicts a covariance and factorizes it
per sample, which is cubic; comparing only against that would be choosing a weak
opponent.

The efficiency argument therefore rests on memory and on the pair-keyed
parameterization, not on wall-clock. An exact likelihood must materialize a
per-sample p by p object however the solve is arranged, and cannot allocate a
single step past p = 4096 on a 15 GB machine, while a pair-keyed head with a
fixed pair budget is flat in memory from p = 512 through 16,384. That gap is a
property of the parameterization and no implementation escapes it.

## Concurrent appends corrupted a results file

Running sweeps as several background processes writing to one CSV interleaved a
header into the middle of the file, which then failed to parse and silently
dropped rows on read. `studies/_io.py` now takes an exclusive lock and writes the
header only when the file is empty.

## Two things to know when reading the real-data numbers

Positive-definiteness on real predictions is not rare, it is universal: 99.9
percent of the 24,001 predicted correlation matrices are indefinite before
projection. The simulations show this is a finite-sample artifact that resolves
along either axis, falling from 99.5 percent at 100 conditions to 0.3 percent at
6400, and LINCS sits at the starved end of both.

DTR-Bench requires batch correction. Drug representations come from the
small-molecule model and target representations from the shRNA model, so the two
carry a strong inter-dataset batch effect. Without removing leading principal
components from the combined matrix, both the correlation and precision
representations score near chance, 0.559 and 0.545, which reads as total failure.
With it they reach 0.567 and 0.590. Note that the shipped procedure selects the
component count by evaluation AUROC, which is selection on the test metric; it
keeps a comparison between arms fair but makes absolute values optimistic.
