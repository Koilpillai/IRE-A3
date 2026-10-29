| dataset | index | QPS @ recall 0.90 | recall reached | QPS @ recall 0.95 | recall reached | setting at 0.95 |
|---|---|---|---|---|---|---|
| sift | Exact (Flat) | 38.8 | 1.000 | 38.8 | 1.000 | - |
| sift | HNSW | 16601.0 | 0.917 | 11521.4 | 0.962 | M=32, efSearch=40 |
| sift | IVF-Flat | 3030.7 | 0.932 | 2148.0 | 0.962 | nlist=4096, nprobe=48 |
| sift | IVF-PQ | 1338.6 | 0.915 | 563.7 | 0.952 | nlist=1024, m=128, nprobe=32 |
| glove | Exact (Flat) | 43.4 | 1.000 | 43.4 | 1.000 | - |
| glove | HNSW | 2218.0 | 0.906 | 984.0 | 0.954 | M=32, efSearch=384 |
| glove | IVF-Flat | 861.0 | 0.916 | 444.8 | 0.952 | nlist=4096, nprobe=384 |
| glove | IVF-PQ | 269.5 | 0.916 | 74.2 | 0.955 | nlist=1024, m=100, nprobe=256 |

Settings are chosen on the tune queries; QPS and recall are measured on the report queries (median of 3), so the recall reached can fall slightly below the target and the plotted curves are not perfectly monotone.
