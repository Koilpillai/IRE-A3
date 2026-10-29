## Part 2: benchmark of HNSW, IVF-Flat and IVF-PQ

Figures: `results/figures/part2_recall_vs_qps_sift.pdf` and `..._glove.pdf`. Tables: `results/tables/part2_*.md`. All speeds are
one query at a time on one thread (median of 3 repeats, report queries 5,000-9,999); the settings shown were chosen on the
tune queries 0-4,999 only. Machine and library versions are in `results/environment.json`.

**Queries per second at recall 0.90 and 0.95** (recall is the one measured on the report queries, so it can land slightly below the target):

| dataset | index | QPS @ 0.90 | recall | QPS @ 0.95 | recall | setting at 0.95 |
|---|---|---|---|---|---|---|
| SIFT | Exact (Flat) | 38.8 | 1.000 | 38.8 | 1.000 | - |
| SIFT | HNSW | 16601 | 0.917 | 11521 | 0.962 | M=32, efSearch=40 |
| SIFT | IVF-Flat | 3031 | 0.932 | 2148 | 0.962 | nlist=4096, nprobe=48 |
| SIFT | IVF-PQ | 1339 | 0.915 | 564 | 0.952 | nlist=1024, m=128, nprobe=32 |
| GloVe-100 | Exact (Flat) | 43.4 | 1.000 | 43.4 | 1.000 | - |
| GloVe-100 | HNSW | 2218 | 0.906 | 984 | 0.954 | M=32, efSearch=384 |
| GloVe-100 | IVF-Flat | 861 | 0.916 | 445 | 0.952 | nlist=4096, nprobe=384 |
| GloVe-100 | IVF-PQ | 270 | 0.916 | 74 | 0.955 | nlist=1024, m=100, nprobe=256 |

**Cost of the index that reaches 0.95** (build time uses 10 threads; size is the serialised index):

| dataset | index | build setting | build time (s) | size (MiB) | bytes per vector |
|---|---|---|---|---|---|
| SIFT | Exact (Flat) | - | 0.2 | 488 | 512 |
| SIFT | HNSW | M=32 | 86 | 752 | 788 |
| SIFT | IVF-Flat | nlist=4096 | 113 | 498 | 522 |
| SIFT | IVF-PQ | nlist=1024, m=128 | 73 | 130 | 137 |
| GloVe-100 | Exact (Flat) | - | 0.2 | 451 | 400 |
| GloVe-100 | HNSW | M=32 | 204 | 763 | 676 |
| GloVe-100 | IVF-Flat | nlist=4096 | 93 | 462 | 409 |
| GloVe-100 | IVF-PQ | nlist=1024, m=100 | 64 | 122 | 108 |

**What the numbers say.** At recall 0.95 HNSW is the fastest on both datasets: 297x (SIFT) and 23x (GloVe) faster than exact
search, against 55x and 10x for IVF-Flat and 15x and 1.7x for IVF-PQ. The gap to IVF-Flat is wider on SIFT (5.4x) than on GloVe
(2.2x), and GloVe needs far more search effort for the same recall (efSearch 384 against 40, nprobe 384 against 48), so GloVe
is the harder dataset for every index. I did not measure why; a likely reason is that GloVe's nearest neighbours are
barely closer than its other points, which the Part 3 query-difficulty experiment could check.

**Why an index fails to reach high recall.** Only IVF-PQ fails to reach 0.95 for most of its settings. With 2 or more
dimensions per code byte (SIFT m <= 64, GloVe m <= 50) its recall tops out at 0.88 (SIFT) and 0.87 (GloVe), and with 4 or
more dimensions per byte at 0.73 or lower (SIFT) and 0.53 or lower (GloVe). This is not a search-effort problem: scanning
every list (nprobe = nlist) gives the same recall (`part2_recall_ceilings.md`: 0.872 against 0.881 for SIFT m=64, 0.867
against 0.868 for GloVe m=50). The distance the index ranks by is computed from the 8-bit PQ codes, and the error of that
approximation is larger than the gap between the 10th neighbour and the points just behind it, so true neighbours are
ranked below others however many lists are scanned. Only one dimension per byte (m = d, four times smaller than raw
floats) gets past 0.95, and even then the ceiling is 0.97. Without re-ranking against the stored vectors there is no way
back; that comparison is the Part 3 "compression" option. HNSW with M=8 on GloVe also did not reach 0.95 (0.949 at efSearch
4096, the largest value tried, and still rising), whereas M=16 and M=32 reached 0.99; I did not test why, a sparser graph
simply needed a far wider beam.

**Caveats.** Three build settings per index (four to five for IVF-PQ) were tried, so a better setting may exist. The
frontier is chosen on the tune half but drawn with report-half numbers, and the report half is slightly easier (higher
recall at the same setting), so the curves are not perfectly monotone and the recall reached at "0.90" ranges from 0.906 to 0.932. QPS is
single-thread, one-at-a-time, so it favours indexes with small per-query overhead and says nothing about batched or
multi-threaded throughput. I explain IVF-PQ's low speed (even though its codes are 4x smaller, it is slower than IVF-Flat) by the per-list lookup tables and
byte-wise table lookups that IVF-Flat's contiguous SIMD distance computation avoids; I did not profile this.
