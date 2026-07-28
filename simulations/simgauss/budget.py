"""Memory sizing for the cost and memory studies.

The dominant allocation is a batch of per-sample dense matrices, costing
batch * p * p * itemsize, and autograd keeps several live at once. These helpers
pick a batch that keeps one tensor inside a stated budget so a sweep cannot take
the machine down.
"""

TENSOR_BUDGET_BYTES = 500_000_000
LIVE_TENSOR_ESTIMATE = 8


def safe_batch(p, budget_bytes=TENSOR_BUDGET_BYTES, itemsize=4, max_batch=256):
    return max(1, min(max_batch, budget_bytes // (itemsize * p * p)))


def projected_peak_gb(batch, p, itemsize=4, live=LIVE_TENSOR_ESTIMATE):
    return live * batch * p * p * itemsize / 1e9


def check(p, batch, ceiling_gb=8.0):
    peak = projected_peak_gb(batch, p)
    if peak > ceiling_gb:
        raise MemoryError(
            f"p={p} batch={batch} projects to {peak:.1f} GB, over the {ceiling_gb} GB "
            f"ceiling. Use batch={safe_batch(p)} instead."
        )
    return peak
