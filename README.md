# CS4.406 Assignment 3 — Benchmarking approximate nearest neighbour search

Status: Parts 1 (exact baseline), 2 (benchmark of three indexes), 3 (compression experiment), 4 (drifting data) and 5 (the report) are implemented.

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
design), Part 2 about 3-4 hours and Part 3 about 25 minutes, Part 4 about 4 minutes; `--quick` takes about 13 minutes in total (measured: 13 min 1 s for the whole of `run_all.sh --quick` on the
development machine, a fast laptop CPU; a slower machine may need longer than the 15 minutes the assignment allows). Run the tests alone with `python -m pytest`.

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

## Part 3

The experiment is the "Compression" option: on SIFT, vary the number of PQ sub-vectors and compare plain IVF-PQ with IVF-PQ
followed by exact re-ranking of the top 100 candidates. The hypothesis is in `part3_hypothesis.md`; it was committed
(`c8c6adc`) before any Part 3 code or result existed, and the answers are in `part3.md`.

`part3_benchmark.py` builds IVF-PQ (`nlist` = 1024, the same as in Part 2) for m in {16, 32, 64, 128}. Each index is
searched in two ways that share the same codes and the same `nprobe`: plain, and wrapped in FAISS `IndexRefineFlat` (top 100
re-ranked against the original vectors, which stay in memory and are counted in the index size). `nprobe` is swept as in Part 2
(capped at 256; `ceiling_check` rows give the recall at `nprobe` = `nlist`). The `candidate_recall` rows give the share of
the true top 10 inside the PQ top 100 (1,000 tune queries, batch search, not timed), the most re-ranking could recover.
It reuses the `sweep` and `measure` functions of `part2_benchmark.py`, so the timing method is identical. `--resume`
skips values of m that are finished.

`part3_report.py` makes `results/figures/part3_compression.png/pdf` (recall against speed; memory against the recall
reached) and `results/tables/part3_*.md/csv`. Settings for the 0.90 and 0.95 targets are chosen on the tune half, numbers
come from the report half, as in Part 2. The IVF-Flat reference row is Part 2's sweep with the same `nlist`.

## Part 4

The question is what happens to each index when the data drifts: it is built on old vectors, then vectors and queries from a
shifted distribution arrive. The hypothesis is in `part4_hypothesis.md` (committed, `7eb8005`, before any Part 4 code or
result existed); the answers are in `part4.md`.

`annbench/drift.py` is the seeded data generator: 100 Gaussian clusters in 64 dimensions (cluster width 1, centres
spread 2 per dimension), 200,000 old vectors, 100,000 new vectors and 2,000 queries (the first 1,000 are the tune half, the last 1,000
the report half); `--quick` uses 20,000 old and 10,000 new vectors. Everything random is drawn once from one seed and the shift is added
afterwards to the new vectors and the queries (along one fixed random direction, in units of 8, the typical distance
of a vector to its cluster centre; shifts 0 (the control), 1, 2 and 4), so the shift is the only thing that changes.

`part4_benchmark.py` runs HNSW (M 16), IVF-Flat (`nlist` 512) and IVF-PQ (`nlist` 512, m 64: the only m of 16, 32 and 64 that
reaches recall 0.95 in the control). Per shift and index it measures two versions on the same data and queries: **stale**
(built on the old vectors, the new ones added without retraining) and **retrained** (built from scratch on all vectors).
The search setting is the smallest `nprobe` / `efSearch` with recall >= 0.95 on the tune queries of the control (shift 0,
stale index; `choose_setting` rows) and is then kept fixed. Recall and speed (one query at a time, one thread, median of
15 repeats, because each pass takes only about 0.1 s) are measured on the report half. The notes of each row hold the shift, the
build / add times and the diagnostics: vectors scanned per query and list sizes (IVF), and the squared error of the PQ codes on
new vectors (IVF-PQ). `part4_report.py` makes `results/figures/part4_drift.png/pdf` and `results/tables/part4_*.md/csv`.
HNSW builds are multi-threaded and not deterministic, so HNSW recall can differ by a few points between two identical builds.

`results/earlier_run_part4_3repeats.csv` holds the Part 4 rows of a first run with 3 timing repeats, which I replaced by the 15-repeat run in
`results.csv`. It is kept only as evidence for the statements about run-to-run variation (HNSW recall, speed); `run_all.sh` does not regenerate it.

## Part 5

`part5_report.py` builds `report.pdf` (at most 5 pages; it stops with an error if it is longer) with reportlab. The tables and figures are
read from `results/results.csv`, `results/tables/` and `results/figures/`, so they are regenerated with everything else. The sentences around them
are written by hand and quote the numbers of the full run: if you re-run the benchmarks, the wall-clock numbers will move a little and the text
will not follow. With `--quick` it writes `results_quick/report.pdf` with the quick numbers in the tables and a note on page 1. Use
`python part5_report.py --author "Name, roll number"` to print a name under the title.

## Layout

| path | contents |
|---|---|
| `annbench/` | `recall.py` recall@k, `timing.py` timing, `exact.py` FAISS exact search, `indexes.py` the three indexes and the re-ranked IVF-PQ, `drift.py` the Part 4 data generator, `analysis.py` sweep stopping / frontier / selection, `data.py` HDF5 loading, `results.py` results.csv, `env.py` machine/versions |
| `part1_exact.py` | Part 1 |
| `part1.md`, `part2.md`, `part3.md`, `part4.md` | the longer written answers for Parts 1 to 4 that the report was trimmed from |
| `hypothesis.md` | the Part 3 and Part 4 hypotheses in one file (the submission copy) |
| `part3_hypothesis.md`, `part4_hypothesis.md` | the same two hypotheses as first committed, each before its experiment was run (see `git log`) |
| `report.md` | a shorter Markdown version of the report; `report.pdf` is the submitted one |
| `ai_log/` | `ai_statement.md` (which file is human-written, AI-assisted or AI-generated) and `prompts.md` (the prompt log) |
| `part2_benchmark.py`, `part2_report.py` | Part 2: run the sweeps; make the figures and tables |
| `part3_benchmark.py`, `part3_report.py` | Part 3: run the experiment; make the figure and tables |
| `part4_benchmark.py`, `part4_report.py` | Part 4: run the experiment; make the figure and tables |
| `part5_report.py` | Part 5: builds `report.pdf` from the tables and figures |
| `tests/` | unit tests (recall function, exact search, indexes, sweep analysis, re-ranking, drift generator) |
| `run_all.sh` | one-command reproduction |
| `results/`, `results_quick/` | generated: `results.csv`, `environment.json`, `figures/`, `tables/` |
