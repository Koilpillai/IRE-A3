## Part 4: drifting data (my own synthetic data)

Hypothesis: `part4_hypothesis.md`, committed before any Part 4 code or result existed (commit `7eb8005`). Generator:
`annbench/drift.py`. Figure: `results/figures/part4_drift.pdf`. Tables: `results/tables/part4_*.md`.

**What I changed.** One thing: the shift of the data that arrives after the index was built. The data are 100 Gaussian clusters
in 64 dimensions. HNSW, IVF-Flat and IVF-PQ are built on 200,000 old vectors; then 100,000 new vectors are added, and 2,000
queries (1,000 for tuning, 1,000 for the numbers) are drawn like the new vectors. The new vectors and the queries are
moved along one fixed direction by 0 (the control), 1, 2 or 4 times a typical vector's distance to its own cluster centre. Everything else
(the draws, the sizes, the build settings) is identical. Each index keeps one search setting, chosen on the control so that recall is about
0.95 (`efSearch` 64, `nprobe` 6, `nprobe` 6). Each is measured **stale** (new vectors just added) and **retrained** (rebuilt on all 300,000).

| index | shift | recall, stale | recall, retrained | speed, stale / retrained |
|---|---|---|---|---|
| HNSW | 0 / 1 / 2 / 4 | 0.956 / 0.944 / 0.966 / 0.989 | 0.951 / 0.959 / 0.932 / 0.975 | 1.00 / 0.92 / 0.90 / 0.93 |
| IVF-Flat | 0 / 1 / 2 / 4 | 0.982 / 0.995 / 1.000 / 0.975 | 0.973 / 1.000 / 1.000 / 1.000 | 0.91 / 0.88 / 0.91 / 0.67 |
| IVF-PQ | 0 / 1 / 2 / 4 | 0.956 / 0.889 / 0.635 / 0.309 | 0.947 / 0.970 / 0.965 / 0.961 | 1.04 / 0.96 / 0.96 / 1.03 |

**What happened.** IVF-PQ collapses: with the same search setting its recall falls from 0.96 to 0.89, 0.64 and 0.31, and retraining brings it
back to 0.96 to 0.97. IVF-Flat keeps its recall until the largest shift (0.975 against 1.000 retrained) but is then about 30%
slower than the retrained index. HNSW is not hurt in recall; the stale graph is about 10% slower than the rebuilt one at every shift above 0.
Recall and speed also move with the shift for the retrained indexes, because the data change (for example, the retrained indexes scan
5,500 instead of 3,800 vectors per query at shift 4), so each stale result has to be read next to its retrained one.

**Mechanism.** For IVF-PQ the cause is measured. The codebook was learned on residuals of the old vectors, and a shifted vector is far from its list
centre, so its code is bad: the squared error of the code on the new vectors is 0.022, 0.46, 12 and 220 at shifts 0, 1, 2 and 4 for the stale
index, against 0.022 to 0.052 after retraining. Wrong distances put the wrong vectors in the top 10, and no search setting fixes that. For IVF-Flat I
expected a few oversized lists and a longer scan, but this is not what the data show: stale and retrained indexes have similar largest lists (3,085 against 2,979 at shift 4) and scan similar numbers of
vectors (5,139 against 5,456). So I do not know why the stale IVF-Flat is slower; I only measured that it is. A likely reason for its small recall loss is that
the queries are far from all the old centroids, so ranking the lists by centroid distance tells little (not measured).
For HNSW I have no mechanism: it has nothing trained in advance and new vectors link to their nearest neighbours when added, which fits the lack of damage.

**Did it support the hypothesis?** Partly. Supported: IVF-PQ is hit hardest and retraining fixes it; HNSW is least affected; the control
stays the same (stale and retrained agree at shift 0 within 0.01 in recall for every index). Not supported: IVF-Flat did not become
"several times slower" (about 1.5 times slower, at shift 4 only, with a small recall loss), and the oversized-list explanation failed. The cost
prediction was wrong at this size: retraining IVF-Flat took 1.2 s and rebuilding HNSW 9.5 s (not minutes); IVF-PQ was the most expensive, 22.9 s, because it
trains codebooks.

**Not tested, and how far to trust this.** Gradual drift, deletions, rotations, new clusters, other dimensions or sizes, real data, other `nlist` or
`M`, other seeds. HNSW graphs differ between runs: an earlier run of the same experiment (3 timing repeats, since replaced because each
timing was too short) gave HNSW recalls that differed by up to 0.04 for the same build (for example 0.916 against 0.959 for the rebuilt index at shift 1); its IVF recalls were identical. So HNSW
recall differences below about 0.04 mean nothing here. Speeds from single runs of 1,000 queries vary by about 10%; only the
IVF-Flat slowdown at shift 4 (0.67 here, 0.72 in the earlier run) and the HNSW gap (0.90 to 0.93, 0.85 to 0.89 earlier) appeared in both runs.
