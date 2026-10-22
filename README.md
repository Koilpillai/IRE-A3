# CS4.406 Assignment 3 — Benchmarking approximate nearest neighbour search

Status: Part 1 (exact baseline) is implemented. Parts 2–5 are not yet.

## Setup

Python 3.11, then `pip install -r requirements.txt`.

Download the two ann-benchmarks files into `data/` (they are not part of the submission):

- http://ann-benchmarks.com/sift-128-euclidean.hdf5
- http://ann-benchmarks.com/glove-100-angular.hdf5

## Reproduce

```
bash run_all.sh            # full datasets, writes results/
bash run_all.sh --quick    # first 100,000 vectors, 1,000 queries per half, writes results_quick/
```

Set `PYTHON=/path/to/python` if `python` is not the interpreter with the requirements installed.
Run on an otherwise idle machine: every number in `results.csv` is a wall-clock timing.
The machine and library versions of a run are saved next to its results in `environment.json`.

Part 1 on the full data takes about 30 minutes (one-query-at-a-time exact search over 1M vectors is slow
by design); `--quick` takes under a minute. Run the tests alone with `python -m pytest`.

## Conventions (all parts)

- Euclidean distance everywhere. GloVe vectors *and* queries are normalised to unit length; the file's own
  distances are cosine distances, so they are converted with `sqrt(2 * d)` before being compared with ours.
- The 10,000 queries are split in half: tune on the first 5,000, report on the last 5,000.
- `--quick` indexes only the first 100,000 vectors, so the neighbours stored in the files no longer apply;
  the reference is then our own exact search (the file comparison is skipped).
- Recall@10 is computed by `annbench/recall.py` from true distances recomputed from the vectors (PQ distances
  are approximate). A returned vector is correct if it is no farther than the 10th exact neighbour, so ties
  at the 10th neighbour are not mistakes.
- Timings are repeated 3 times; the median is reported. `results.csv` also keeps every repeat.

## Part 1

`part1_exact.py` builds `faiss.IndexFlatL2` and, per dataset:

1. computes the exact top-100 for all 10,000 queries (the reference for later parts);
2. checks it against the neighbours stored in the file (tie-aware recall in both directions, identical
   top-10 id sets, distance differences) and stops if recall is below 0.999;
3. times exact top-10 search over the 5,000 report queries, one query at a time on one thread, as one batch
   on one thread, and as one batch on one thread per physical core (`--threads`, default: the physical core
   count). The single-thread batch is an addition to the assignment's two timings: it separates what batching
   gives from what multi-threading gives. Hyper-threads are not used: on the development machine (WSL2) the
   same batch took 2.4-15 s with 20 threads but 2.9-4.6 s with 10, so 20-thread timings were not repeatable.

## Layout

| path | contents |
|---|---|
| `annbench/` | `recall.py` recall@k, `timing.py` timing, `exact.py` FAISS exact search, `data.py` HDF5 loading, `results.py` results.csv, `env.py` machine/versions |
| `part1_exact.py` | Part 1 |
| `tests/` | unit tests (recall function, exact search) |
| `run_all.sh` | one-command reproduction |
| `results/`, `results_quick/` | generated: `results.csv`, `environment.json` |
