| dataset | index | build setting at 0.95 | build time (s) | index size (MiB) | bytes per vector |
|---|---|---|---|---|---|
| sift | Exact (Flat) | - | 0.2 | 488 | 512 |
| sift | HNSW | {M: 32, efConstruction: 200} | 86.4 | 752 | 788 |
| sift | IVF-Flat | {nlist: 4096} | 113.3 | 498 | 522 |
| sift | IVF-PQ | {nlist: 1024, m: 128} | 73.3 | 130 | 137 |
| glove | Exact (Flat) | - | 0.2 | 451 | 400 |
| glove | HNSW | {M: 32, efConstruction: 200} | 204.4 | 763 | 676 |
| glove | IVF-Flat | {nlist: 4096} | 93.5 | 462 | 409 |
| glove | IVF-PQ | {nlist: 1024, m: 100} | 64.3 | 122 | 108 |

* recall 0.95 was not reached; the setting with the highest recall is shown.
