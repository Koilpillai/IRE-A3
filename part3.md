## Part 3: compression (SIFT, IVF-PQ with and without re-ranking)

Hypothesis: `part3_hypothesis.md`, committed before any Part 3 code or result existed (commit `c8c6adc`).

**Setup.** SIFT, 1,000,000 vectors, IVF-PQ with `nlist` = 1024 and 8-bit codes. The number of PQ sub-vectors `m` is 16, 32, 64 or
128 (8, 4, 2 or 1 dimensions per code byte). For each `m` one index is built, and two versions search the very same codes:
**plain**, and **re-ranked**, which takes the top 100 PQ candidates and re-ranks them with exact distances to the original
vectors (FAISS `IndexRefineFlat`, `k_factor` 10). `nprobe` is swept from 1 to at most 256, as in Part 2 (one query at a time, one
thread, median of 3 on the report queries, settings chosen on the tune queries). Figure: `results/figures/part3_compression.pdf`.
Tables: `results/tables/part3_*.md`. Code: `part3_benchmark.py`, `part3_report.py`.

| m | version | bytes per vector | highest recall (nprobe <= 256) | recall at nprobe = nlist | QPS @ 0.90 | QPS @ 0.95 |
|---|---|---|---|---|---|---|
| 16 | plain | 25 | 0.580 | 0.553 | not reached | not reached |
| 16 | re-ranked | 537 | 0.981 | 0.975 | 5977 | 3676 |
| 32 | plain | 41 | 0.734 | 0.711 | not reached | not reached |
| 32 | re-ranked | 553 | 0.992 | 0.999 | 4383 | 3201 |
| 64 | plain | 73 | 0.881 | 0.872 | not reached | not reached |
| 64 | re-ranked | 585 | 0.993 | 1.000 | 2662 | 1881 |
| 128 | plain | 137 | 0.967 | 0.965 | 1415 | 739 |
| 128 | re-ranked | 649 | 0.993 | 1.000 | 1242 | 930 |
| - | IVF-Flat, nlist 1024 (Part 2) | 521 | 0.993 | - | 2003 | 1367 |

"QPS @ 0.90" and "@ 0.95" are measured at the `nprobe` that first reached the target on the tune queries. Because `nprobe` moves in
steps, the report-half recall can be above the target (0.915 to 0.937 for "0.90", 0.952 to 0.967 for "0.95"), which favours the
versions that were lucky with the grid; read small differences with care.

**Same `nprobe`, plain against re-ranked** (report queries):

| m | median QPS ratio (re-ranked / plain) | lowest | highest | median extra time per query |
|---|---|---|---|---|
| 16 | 0.70 | 0.50 | 0.99 | 35 us |
| 32 | 0.70 | 0.53 | 0.99 | 34 us |
| 64 | 0.84 | 0.51 | 0.95 | 38 us |
| 128 | 0.88 | 0.77 | 0.97 | 47 us |

## Did the results support the hypothesis?

Partly. Three of the five predictions held cleanly (1, 4 and 5), one held but with wrong numbers at the small end (2), and one, about speed, was only half right (3).

1. **Plain IVF-PQ: fewer sub-vectors, lower recall. Supported.** The highest recall is 0.58, 0.73, 0.88 and 0.97 for `m` = 16, 32,
   64 and 128, close to the 0.6 and 0.97 I expected at the two ends. Scanning every list does not help (0.553, 0.711, 0.872 and 0.965
   at `nprobe` = `nlist`), so the limit comes from the compressed distances, not from the search.
2. **Re-ranking helps most when `m` is small. Supported, but my numbers were wrong at `m` = 16.** The gain in the highest recall is
   +0.40, +0.26, +0.11 and +0.03 for `m` = 16, 32, 64 and 128. I expected recall of 0.85 to 0.9 at `m` = 16; it reached 0.981 (and 0.966 already
   at `nprobe` = 32). The reason is that the PQ top 100 still holds most of the true top 10 even with 8 dimensions per byte (96% at
   `nprobe` = 32, against 98% for every `m` of 32 or more). At the same `nprobe` the re-ranked recall is nearly the same for every
   `m` of 32 or more (0.937 at `nprobe` = 16), and so is the share of the true top 10 inside the PQ top 100 (within a thousandth of each other for
   `m` of 32 or more). This suggests that once the true neighbours are in the lists that are probed, the PQ top 100 holds nearly all of them, so
   what is left is a matter of which lists were probed, not of the code size (I did not measure the share of neighbours inside the probed lists directly).
3. **Re-ranking costs some speed, but is faster at equal recall. Half right.** At the same `nprobe`, re-ranking cut speed by a median of
   12% to 30% (ratio 0.70 to 0.88), inside my 10% to 30%, but the cut reached 50% at small `nprobe`, which is worse than I expected. The
   extra time per query is roughly the same whatever `m` is (a median of 34 to 47 microseconds), which is what a fixed cost of
   reading and comparing 100 original vectors would look like (I did not profile it), and a fixed cost weighs more when the PQ scan is short:
   small `nprobe` and small `m`. At equal recall, re-ranking won only where the plain index could reach the target at all, and
   only at 0.95: at `m` = 128 it ran at 930 QPS against 739 (+26%), because it needed `nprobe` 24 instead of 32. At 0.90 it was
   slower (1242 against 1415 QPS at `m` = 128, −12%), since both needed `nprobe` 16 and re-ranking added its cost. This is one of the
   cases I wrote down as showing me wrong. For `m` below 128 the plain index never reaches 0.90 or 0.95, so there is nothing to compare.
4. **Fewer sub-vectors scan faster. Supported.** Plain speed at `nprobe` = 16 is 7990, 4961, 2937 and 1415 QPS for `m` = 16, 32, 64
   and 128: it roughly halves each time `m` doubles.
5. **Re-ranking does not save memory. Supported, and exactly as computed.** Plain IVF-PQ takes `m` + 9 bytes per vector
   (25, 41, 73, 137) and the re-ranked version takes 512 more (537, 553, 585, 649), which is more than IVF-Flat's 521. What re-ranking gives is recall, and the original
   vectors have to stay in RAM to do it.

**Something I did not predict.** Re-ranked IVF-PQ with a *small* `m` is the fastest of everything here at recall 0.95: 3676 QPS at `m` = 16,
against 1367 for IVF-Flat with the same `nlist` (2.7x) and 739 for the best plain IVF-PQ (5x), at about the same memory as
IVF-Flat (537 against 521 bytes per vector). I think the reason is that the scan reads 16 bytes per candidate instead of 512, and the 100
exact distances are cheap next to it, but I did not measure that. The comparison with IVF-Flat uses the Part 2 run, timed in an earlier session on the same machine.

**What it means.** Without re-ranking, compression buys a 4x to 20x smaller index and costs recall that no search effort can win back. With
re-ranking, the recall is back, but the memory saving is gone unless the original vectors can be kept somewhere other than RAM,
which I did not test. The small codes are then useful only for speed.

**Not tested, or to be careful about.** Only one `nlist` (1024), one re-ranking depth (100) and one dataset (SIFT, which has
small integer-valued vectors; GloVe may behave differently). One timing session per setting. The first (tune-half) timing of each sweep
can be a cold start (for example 1,805 QPS for re-ranked `m` = 128 at `nprobe` = 1, against 9,991 on the report half); it only affects
the choice of `nprobe` at recall far below 0.90. The share of true neighbours inside the PQ top 100 is measured on 1,000 tune
queries, while recalls come from 5,000 or from the report half, so they only agree to about a percentage point.
