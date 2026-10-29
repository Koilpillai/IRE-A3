# CS4.406 Assignment 3 — Benchmarking approximate nearest neighbour search

Status: Parts 1 (exact baseline) and 2 (benchmark of three indexes) are implemented. Parts 3–5 are not yet.

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

On the full data Part 1 takes about 35 minutes (one-query-at-a-time exact search over 1M vectors is slow by
design) and Part 2 about 3-4 hours; `--quick` takes about 8 minutes in total (measured: 8 min 19 s on the development machine). Run the tests alone with `python -m pytest`.

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

## Part 2

`part2_benchmark.py` benchmarks HNSW (hnswlib), IVF-Flat and IVF-PQ (FAISS) on both datasets. Build settings
(`annbench/indexes.py: build_settings`): HNSW `M` in {8, 16, 32} with efConstruction 200; IVF-Flat `nlist` in
{256, 1024, 4096}; IVF-PQ (8-bit codes, no re-ranking) with `nlist` 1024 and m in {16, 32, 64, 128} (SIFT) or
{10, 20, 50, 100} (GloVe), plus `nlist` 4096 with the middle m. For each, the search parameter (efSearch or nprobe) is
swept upwards until recall reaches 0.99 or stops improving. Speed is one query at a time on one thread, on both
query halves: the tune half timed once, the report half timed 3 times (median). Indexes are built with one thread
per physical core, so build times depend on the machine's core count. Index size is the size of the serialised index.
For IVF-PQ, one extra row per build setting (`ceiling_check`) gives the recall when every list is scanned.

`part2_report.py` then chooses, **on the tune half only**, the points of each index's best trade-off curve (Pareto
frontier over all build settings and search values) and the setting that is fastest at recall 0.90 and 0.95, and
shows the **report-half** numbers for them: `results/figures/part2_recall_vs_qps_*.png/pdf` (recall on x, speed on
y as in ann-benchmarks) and `results/tables/part2_*.md/csv`. Because of this split, the recall reached on the
report half can be slightly below 0.90/0.95, and the plotted curves are not perfectly monotone.
`--resume` lets `part2_benchmark.py` continue after an interruption, skipping finished build settings.

If `import hnswlib` fails with `GLIBCXX_3.4.xx not found` (Anaconda-based Python), `run_all.sh` preloads the system
libstdc++ automatically; when running the scripts by hand use
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python part2_benchmark.py`.

## Layout

| path | contents |
|---|---|
| `annbench/` | `recall.py` recall@k, `timing.py` timing, `exact.py` FAISS exact search, `indexes.py` the three indexes, `analysis.py` sweep stopping / frontier / selection, `data.py` HDF5 loading, `results.py` results.csv, `env.py` machine/versions |
| `part1_exact.py` | Part 1 |
| `part1.md`, `part2.md` | written answers for Parts 1 and 2 (to go into the report) |
| `part2_benchmark.py`, `part2_report.py` | Part 2: run the sweeps; make the figures and tables |
| `tests/` | unit tests (recall function, exact search, indexes, sweep analysis) |
| `run_all.sh` | one-command reproduction |
| `results/`, `results_quick/` | generated: `results.csv`, `environment.json`, `figures/`, `tables/` |
