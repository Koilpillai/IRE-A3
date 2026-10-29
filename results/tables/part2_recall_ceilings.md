| dataset | index | build setting | highest recall in sweep | at | recall with nprobe=nlist (500 queries) |
|---|---|---|---|---|---|
| sift | HNSW | {M: 16, efConstruction: 200} | 0.994 | {efSearch: 160} | - |
| sift | HNSW | {M: 32, efConstruction: 200} | 0.993 | {efSearch: 96} | - |
| sift | HNSW | {M: 8, efConstruction: 200} | 0.993 | {efSearch: 384} | - |
| sift | IVF-Flat | {nlist: 1024} | 0.993 | {nprobe: 48} | - |
| sift | IVF-Flat | {nlist: 256} | 0.995 | {nprobe: 24} | - |
| sift | IVF-Flat | {nlist: 4096} | 0.994 | {nprobe: 128} | - |
| sift | IVF-PQ | {nlist: 1024, m: 128} | 0.967 | {nprobe: 192} | 0.965 |
| sift | IVF-PQ | {nlist: 1024, m: 16} | 0.580 | {nprobe: 64} | 0.553 |
| sift | IVF-PQ | {nlist: 1024, m: 32} | 0.734 | {nprobe: 96} | 0.711 |
| sift | IVF-PQ | {nlist: 1024, m: 64} | 0.881 | {nprobe: 128} | 0.872 |
| sift | IVF-PQ | {nlist: 4096, m: 32} | 0.747 | {nprobe: 192} | 0.733 |
| glove | HNSW | {M: 16, efConstruction: 200} | 0.991 | {efSearch: 4096} | - |
| glove | HNSW | {M: 32, efConstruction: 200} | 0.992 | {efSearch: 1536} | - |
| glove | HNSW | {M: 8, efConstruction: 200} | 0.950 | {efSearch: 4096} | - |
| glove | IVF-Flat | {nlist: 1024} | 0.992 | {nprobe: 384} | - |
| glove | IVF-Flat | {nlist: 256} | 0.994 | {nprobe: 128} | - |
| glove | IVF-Flat | {nlist: 4096} | 0.995 | {nprobe: 1536} | - |
| glove | IVF-PQ | {nlist: 1024, m: 100} | 0.970 | {nprobe: 1024} | 0.965 |
| glove | IVF-PQ | {nlist: 1024, m: 10} | 0.306 | {nprobe: 64} | 0.309 |
| glove | IVF-PQ | {nlist: 1024, m: 20} | 0.527 | {nprobe: 384} | 0.516 |
| glove | IVF-PQ | {nlist: 1024, m: 50} | 0.868 | {nprobe: 768} | 0.867 |
| glove | IVF-PQ | {nlist: 4096, m: 20} | 0.542 | {nprobe: 768} | 0.544 |
