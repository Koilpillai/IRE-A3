| index | shift | version | recall@10 | QPS | spread of the repeats | QPS / QPS of the stale index at shift 0 | QPS / QPS of the retrained index at the same shift | vectors scanned per query | largest list | PQ squared error on new vectors |
|---|---|---|---|---|---|---|---|---|---|---|
| HNSW | 0 | stale | 0.956 | 14707 | 6% | 1.00 | 1.00 | - | - | - |
| HNSW | 0 | retrained | 0.951 | 14758 | 10% | 1.00 | 1.00 | - | - | - |
| HNSW | 1 | stale | 0.944 | 18280 | 29% | 1.24 | 0.92 | - | - | - |
| HNSW | 1 | retrained | 0.959 | 19832 | 13% | 1.35 | 1.00 | - | - | - |
| HNSW | 2 | stale | 0.966 | 19352 | 80% | 1.32 | 0.90 | - | - | - |
| HNSW | 2 | retrained | 0.932 | 21445 | 11% | 1.46 | 1.00 | - | - | - |
| HNSW | 4 | stale | 0.989 | 19281 | 42% | 1.31 | 0.93 | - | - | - |
| HNSW | 4 | retrained | 0.975 | 20781 | 16% | 1.41 | 1.00 | - | - | - |
| IVF-Flat | 0 | stale | 0.982 | 15440 | 23% | 1.00 | 0.91 | 3705 | 1659 | - |
| IVF-Flat | 0 | retrained | 0.973 | 16980 | 11% | 1.10 | 1.00 | 3814 | 3034 | - |
| IVF-Flat | 1 | stale | 0.995 | 15483 | 35% | 1.00 | 0.88 | 3766 | 2029 | - |
| IVF-Flat | 1 | retrained | 1.000 | 17508 | 16% | 1.13 | 1.00 | 3600 | 3034 | - |
| IVF-Flat | 2 | stale | 1.000 | 14777 | 29% | 0.96 | 0.91 | 3837 | 2110 | - |
| IVF-Flat | 2 | retrained | 1.000 | 16178 | 24% | 1.05 | 1.00 | 3831 | 2923 | - |
| IVF-Flat | 4 | stale | 0.975 | 11913 | 27% | 0.77 | 0.67 | 5139 | 3085 | - |
| IVF-Flat | 4 | retrained | 1.000 | 17663 | 32% | 1.14 | 1.00 | 5456 | 2979 | - |
| IVF-PQ | 0 | stale | 0.956 | 10326 | 13% | 1.00 | 1.04 | 3705 | 1659 | 0.0218 |
| IVF-PQ | 0 | retrained | 0.947 | 9901 | 22% | 0.96 | 1.00 | 3814 | 3034 | 0.0217 |
| IVF-PQ | 1 | stale | 0.889 | 10130 | 40% | 0.98 | 0.96 | 3766 | 2029 | 0.462 |
| IVF-PQ | 1 | retrained | 0.970 | 10581 | 13% | 1.02 | 1.00 | 3600 | 3034 | 0.0253 |
| IVF-PQ | 2 | stale | 0.635 | 10100 | 31% | 0.98 | 0.96 | 3837 | 2110 | 12 |
| IVF-PQ | 2 | retrained | 0.965 | 10483 | 18% | 1.02 | 1.00 | 3831 | 2923 | 0.0311 |
| IVF-PQ | 4 | stale | 0.309 | 8823 | 38% | 0.85 | 1.03 | 5139 | 3085 | 220 |
| IVF-PQ | 4 | retrained | 0.961 | 8529 | 11% | 0.83 | 1.00 | 5456 | 2979 | 0.0522 |

Report queries, one at a time on one thread (median of 15 repeats; 'spread' is (slowest - fastest) / median of those repeats). Shift 0 is the control. A shift is measured in units of a typical vector's distance to its own cluster centre. Part of a speed change from shift 0 comes from the data itself (the same change shows in the retrained version); the column 'QPS / QPS of the retrained index at the same shift' isolates the effect of staleness.
