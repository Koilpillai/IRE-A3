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
