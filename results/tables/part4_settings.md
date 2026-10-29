| index | build setting | search setting (chosen at shift 0) | tune recall |
|---|---|---|---|
| HNSW | M=16, efConstruction=200 | efSearch=64 | 0.952 |
| IVF-Flat | nlist=512 | nprobe=6 | 0.978 |
| IVF-PQ | nlist=512, m=64 | nprobe=6 | 0.954 |

The search setting is the smallest that reaches recall 0.95 on the tune queries of the control, measured with the stale index at shift 0; it is then kept fixed.
