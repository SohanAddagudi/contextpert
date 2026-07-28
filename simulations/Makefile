# One target per table in the paper. Runtimes are for 4 CPU cores, no GPU.
PY ?= ../.venv/bin/python

.PHONY: all test recovery cost memory scale tables clean

all: recovery cost memory scale tables    ## everything, roughly 70 minutes

test:                                  ## analytic checks on the matrix routines
	$(PY) -m pytest tests/ -q

recovery:                              ## ~25 min: three sweeps against known ground truth
	$(PY) -m studies.recovery --sweep conditions
	$(PY) -m studies.recovery --sweep allocation
	$(PY) -m studies.recovery --sweep replicates

cost:                                  ## ~10 min: per-step cost vs exact NLL, and pair subsampling
	$(PY) -m studies.cost
	$(PY) -m studies.subsample_cost

memory:                                ## ~15 min: where the exact NLL runs out of memory
	$(PY) -m studies.memory

scale:                                 ## ~20 min: p=20k training and held-out node transfer
	$(PY) -m studies.scale

tables:                                ## regenerate paper tables from committed results
	$(PY) -m studies.summarize

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
