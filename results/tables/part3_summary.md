| m | version | bytes per vector | index size (MiB) | PQ build time (s) | highest recall (nprobe <= 256) | recall at nprobe=nlist | QPS @ 0.90 | recall reached | nprobe | QPS @ 0.95 | recall reached | nprobe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | plain | 25 | 24 | 12.3 | 0.580 | 0.553 | not reached | - | - | not reached | - | - |
| 16 | re-ranked (top 100) | 537 | 512 | 12.3 | 0.981 | 0.975 | 5977 | 0.928 | 16 | 3676 | 0.966 | 32 |
| 32 | plain | 41 | 39 | 13.2 | 0.734 | 0.711 | not reached | - | - | not reached | - | - |
| 32 | re-ranked (top 100) | 553 | 527 | 13.2 | 0.992 | 0.999 | 4383 | 0.937 | 16 | 3201 | 0.967 | 24 |
| 64 | plain | 73 | 69 | 15.3 | 0.881 | 0.872 | not reached | - | - | not reached | - | - |
| 64 | re-ranked (top 100) | 585 | 558 | 15.3 | 0.993 | 1.000 | 2662 | 0.937 | 16 | 1881 | 0.967 | 24 |
| 128 | plain | 137 | 130 | 73.7 | 0.967 | 0.965 | 1415 | 0.915 | 16 | 739 | 0.952 | 32 |
| 128 | re-ranked (top 100) | 649 | 619 | 73.7 | 0.993 | 1.000 | 1242 | 0.937 | 16 | 930 | 0.967 | 24 |
| - | IVF-Flat, nlist=1024 (Part 2, for reference) | 521 | 496 | 9.7 | 0.993 | - | 2003 | 0.937 | 16 | 1367 | 0.967 | 24 |

nprobe is chosen on the tune queries; QPS and recall are measured on the report queries (median of 3). 'recall at nprobe=nlist' uses 500 tune queries. Build time is that of the shared PQ index; wrapping it for re-ranking does not copy the vectors and was not timed.
