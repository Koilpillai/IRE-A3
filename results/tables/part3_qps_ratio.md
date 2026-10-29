| m | median QPS ratio | lowest | highest | median extra time per query (us) | lowest | highest | nprobe values compared |
|---|---|---|---|---|---|---|---|
| 16 | 0.70 | 0.50 | 0.99 | 35 | 5 | 48 | 12 |
| 32 | 0.70 | 0.53 | 0.99 | 34 | 4 | 44 | 11 |
| 64 | 0.84 | 0.51 | 0.95 | 38 | 33 | 47 | 11 |
| 128 | 0.88 | 0.77 | 0.97 | 47 | 18 | 98 | 11 |

QPS of the re-ranked version divided by QPS of the plain one at the same nprobe, over every nprobe of the sweep; the extra time is the difference of the two times per query.
