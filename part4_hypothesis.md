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
