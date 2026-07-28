"""Shared result writing.

Each row is one (study, estimator, setting, seed, metric, value) tuple. Writing
tidy rows rather than wide tables means a study can add a metric without
invalidating existing files, and analysis pivots at read time.

Results are written with an exclusive lock because sweeps are often run as
several concurrent processes; an earlier version interleaved headers mid-file
when two processes appended at once.
"""

import csv
import fcntl
from pathlib import Path

FIELDS = ["study", "estimator", "setting", "n_contexts", "samples_per_context",
          "p", "seed", "metric", "value"]

RESULTS = Path(__file__).resolve().parents[1] / "results"


def write_rows(path, rows, fields=FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
