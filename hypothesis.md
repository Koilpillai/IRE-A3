# Hypotheses

These were written and committed before the experiments were run. The git history shows this:
the Part 3 hypothesis is in commit `c8c6adc` and the Part 4 hypothesis is in commit `7eb8005`,
each before any code or result for that part. They are joined here, unchanged, as one file.

---

# Part 3 hypothesis: compression (SIFT, IVF-PQ with and without re-ranking)

## What I will test

On SIFT (1M vectors), I will change the number of PQ sub-vectors `m` (16, 32, 64, 128) and compare two versions of IVF-PQ:

- **plain**: rank by the PQ codes only;
- **re-ranked**: take the top 100 candidates from the PQ codes, then re-rank them with exact distances.

For both I will measure recall@10, queries per second (one query at a time, one thread) and memory. Everything else stays the same as in Part 2 (`nlist` = 1024, 8-bit codes, tune on the first 5,000 queries, report on the last 5,000).

## What I expect

1. **Plain IVF-PQ: fewer sub-vectors, lower recall.** Each code is smaller, so distances are rougher. From Part 2 I already expect a ceiling of roughly 0.6 at `m` = 16 and roughly 0.97 at `m` = 128, and scanning more lists will not fix it.
2. **Re-ranking will help most when `m` is small.** The PQ ranking is noisy, but the true top 10 should still be somewhere in the PQ top 100. Exact distances then put them back in order. My guess is that recall will pass 0.95 for `m` of 32 and above, and reach somewhere around 0.85 to 0.9 at `m` = 16. For `m` = 128 the gain will be small, because plain PQ is already close to its limit.
3. **Re-ranking costs some speed, but less than it gains.** Re-ranking means reading 100 full vectors per query, which is small next to scanning thousands of codes. I expect QPS at the same `nprobe` to fall by about 10 to 30 percent. But re-ranked IVF-PQ should reach a given recall with fewer lists scanned, so at equal recall it should be faster than plain.
4. **Fewer sub-vectors means faster scanning**, since there are fewer table lookups per code.
5. **Re-ranking will not save memory.** To re-rank, the original 512-byte vectors must be kept. Plain PQ should take about `m` + 9 bytes per vector (73 bytes at `m` = 64 in Part 2). Re-ranked PQ should take that plus 512, which is more than IVF-Flat (522 bytes). So re-ranking buys recall, not memory. It only saves memory if the full vectors are kept on disk, which I will not test.

## What would show I was wrong

- Re-ranking does not lift recall much at small `m`. That would mean the true neighbours are often missing from the PQ top 100.
- Re-ranking costs far more than 30 percent of QPS.
- Re-ranked PQ at the same recall is slower than plain PQ.

## What I am not testing

Other values of `nlist`, other re-ranking depths than 100, 4-bit codes, keeping the full vectors on disk, GloVe, and multi-threaded speed.

---

# Part 4 hypothesis: drifting data (synthetic clusters)

## What I will test

I will build each index on one set of data, then add new vectors and send queries that come from a shifted version of that data. The question is: which index gets worse, and does it recover if I retrain or rebuild?

**Data (made by my own seeded generator, 300,000 vectors at most).** Vectors in 64 dimensions drawn from 100 Gaussian clusters.
- **Old data:** 200,000 vectors. Every index is built on these.
- **New data:** 100,000 vectors from the same clusters, all moved by the same fixed direction. These are added to the index without retraining. The queries (2,000) come from the same shifted distribution.

**The one thing I change is the size of the shift.** I measure it in units of a typical vector's distance to its own cluster centre: 0 (the control, nothing moved), 1, 2 and 4. The number of clusters, the dimension, the sizes and the seed stay the same.

**Indexes:** HNSW, IVF-Flat and IVF-PQ (all three). For each I pick the search setting (`efSearch` or `nprobe`) that gives about 0.95 recall in the control, and keep it fixed. Then I see what happens to recall and speed as the shift grows. I then retrain (IVF) or rebuild (HNSW) on all the data and measure again.

## What I expect

1. **IVF-Flat: slower, not less accurate.** The cluster centres were learned from the old data. Shifted vectors all fall into the few lists nearest to where they landed, so those lists grow far larger than the average one. A query has to scan those fat lists, so speed should fall as the shift grows (I guess several times slower at the largest shift). Recall at the same `nprobe` should stay about the same, because a query and its neighbours move together and land in the same lists. Retraining should spread the lists out again and bring back the control speed.
2. **IVF-PQ: gets worse in both.** It has the same fat-list problem, and its codebook was also learned from the old data. The shifted vectors sit far from their list centre, so the codes describe them badly. I expect recall at the same `nprobe` to fall clearly as the shift grows, and retraining to fix it.
3. **HNSW: least affected.** It has nothing trained in advance. Each new vector links to its nearest neighbours when inserted. I expect little change in recall and speed, maybe a small recall drop at the largest shift, because only a few edges lead from the old region to the new one and the search starts in the old region.
4. **Cost of recovering.** Retraining IVF should take seconds. Rebuilding HNSW from scratch should take minutes, so it should be the most expensive recovery. This only matters if HNSW needs it, which I expect it will not.
5. **The control (shift 0) should be unchanged by adding data.** Any change there would point to a mistake in my set-up and not to drift.

## What would show I was wrong

- IVF-Flat speed does not change with the shift, or its recall drops a lot.
- HNSW gets worse than IVF-Flat.
- Retraining or rebuilding does not bring back the control numbers.
- The control itself changes after adding the new data.

## What I am not testing

Gradual drift over time, deleting vectors, rotations, new clusters appearing, other dimensions or data sizes, real data, and any IVF `nlist` or HNSW `M` other than one fixed choice for each.
