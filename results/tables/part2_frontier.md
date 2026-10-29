| dataset | index | build | search | recall@10 | QPS |
|---|---|---|---|---|---|
| sift | HNSW | {"M": 8, "efConstruction": 200} | {"efSearch": 10} | 0.5865 | 51529.5 |
| sift | HNSW | {"M": 8, "efConstruction": 200} | {"efSearch": 12} | 0.6276 | 47407.3 |
| sift | HNSW | {"M": 8, "efConstruction": 200} | {"efSearch": 20} | 0.7321 | 35545.4 |
| sift | HNSW | {"M": 8, "efConstruction": 200} | {"efSearch": 24} | 0.7653 | 33176.4 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 16} | 0.8141 | 28282.7 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 20} | 0.8493 | 25225.5 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 24} | 0.8760 | 21475.4 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 32} | 0.9121 | 17416.5 |
| sift | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 24} | 0.9167 | 16601.0 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 40} | 0.9341 | 15175.1 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 48} | 0.9482 | 13450.9 |
| sift | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 40} | 0.9624 | 11521.4 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 64} | 0.9673 | 10577.5 |
| sift | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 48} | 0.9721 | 10219.5 |
| sift | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 64} | 0.9831 | 8014.7 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 96} | 0.9841 | 7091.6 |
| sift | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 80} | 0.9891 | 6621.3 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 128} | 0.9910 | 5585.5 |
| sift | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 96} | 0.9926 | 4711.4 |
| sift | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 160} | 0.9944 | 4550.3 |
| sift | IVF-Flat | {"nlist": 1024} | {"nprobe": 1} | 0.3912 | 21784.8 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 2} | 0.4449 | 15781.0 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 3} | 0.5363 | 13692.0 |
| sift | IVF-Flat | {"nlist": 1024} | {"nprobe": 2} | 0.5598 | 11610.8 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 4} | 0.5982 | 10904.7 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 6} | 0.6849 | 10165.3 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 8} | 0.7405 | 8425.4 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 12} | 0.8112 | 6577.2 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 16} | 0.8551 | 5016.6 |
| sift | IVF-Flat | {"nlist": 1024} | {"nprobe": 8} | 0.8510 | 3850.1 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 24} | 0.9037 | 4307.1 |
| sift | IVF-Flat | {"nlist": 1024} | {"nprobe": 12} | 0.9084 | 2500.3 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 32} | 0.9322 | 3030.7 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 48} | 0.9619 | 2148.0 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 64} | 0.9767 | 1869.9 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 96} | 0.9885 | 1261.4 |
| sift | IVF-Flat | {"nlist": 4096} | {"nprobe": 128} | 0.9941 | 923.6 |
| sift | IVF-Flat | {"nlist": 256} | {"nprobe": 24} | 0.9950 | 395.1 |
| sift | IVF-PQ | {"nlist": 1024, "m": 16} | {"nprobe": 2} | 0.4240 | 32270.5 |
| sift | IVF-PQ | {"nlist": 1024, "m": 16} | {"nprobe": 3} | 0.4718 | 25026.2 |
| sift | IVF-PQ | {"nlist": 1024, "m": 32} | {"nprobe": 2} | 0.4952 | 23583.9 |
| sift | IVF-PQ | {"nlist": 1024, "m": 16} | {"nprobe": 4} | 0.5003 | 20932.4 |
| sift | IVF-PQ | {"nlist": 1024, "m": 32} | {"nprobe": 3} | 0.5618 | 16151.2 |
| sift | IVF-PQ | {"nlist": 1024, "m": 32} | {"nprobe": 4} | 0.6042 | 14692.7 |
| sift | IVF-PQ | {"nlist": 1024, "m": 64} | {"nprobe": 3} | 0.6270 | 12134.0 |
| sift | IVF-PQ | {"nlist": 1024, "m": 32} | {"nprobe": 6} | 0.6512 | 10810.4 |
| sift | IVF-PQ | {"nlist": 1024, "m": 64} | {"nprobe": 4} | 0.6828 | 10749.4 |
| sift | IVF-PQ | {"nlist": 1024, "m": 32} | {"nprobe": 8} | 0.6758 | 8780.8 |
| sift | IVF-PQ | {"nlist": 4096, "m": 32} | {"nprobe": 16} | 0.6922 | 7768.7 |
| sift | IVF-PQ | {"nlist": 1024, "m": 64} | {"nprobe": 6} | 0.7492 | 7508.4 |
| sift | IVF-PQ | {"nlist": 1024, "m": 64} | {"nprobe": 8} | 0.7867 | 5869.0 |
| sift | IVF-PQ | {"nlist": 1024, "m": 64} | {"nprobe": 12} | 0.8272 | 4300.5 |
| sift | IVF-PQ | {"nlist": 1024, "m": 64} | {"nprobe": 16} | 0.8464 | 3274.2 |
| sift | IVF-PQ | {"nlist": 1024, "m": 64} | {"nprobe": 24} | 0.8642 | 2212.4 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 12} | 0.8894 | 1764.8 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 16} | 0.9152 | 1338.6 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 24} | 0.9408 | 921.3 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 32} | 0.9519 | 563.7 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 48} | 0.9609 | 372.8 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 64} | 0.9637 | 365.7 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 96} | 0.9657 | 251.9 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 128} | 0.9662 | 179.3 |
| sift | IVF-PQ | {"nlist": 1024, "m": 128} | {"nprobe": 192} | 0.9665 | 128.0 |
| glove | HNSW | {"M": 8, "efConstruction": 200} | {"efSearch": 10} | 0.2999 | 30868.3 |
| glove | HNSW | {"M": 8, "efConstruction": 200} | {"efSearch": 12} | 0.3333 | 31028.9 |
| glove | HNSW | {"M": 8, "efConstruction": 200} | {"efSearch": 16} | 0.3828 | 23678.6 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 10} | 0.4720 | 26221.8 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 12} | 0.5054 | 23720.4 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 16} | 0.5584 | 19599.5 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 10} | 0.5755 | 12451.2 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 20} | 0.5977 | 15913.7 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 24} | 0.6277 | 15154.9 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 16} | 0.6526 | 12814.5 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 32} | 0.6711 | 12216.0 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 20} | 0.6882 | 11667.0 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 40} | 0.7031 | 10334.8 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 24} | 0.7165 | 9778.7 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 48} | 0.7291 | 9216.7 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 32} | 0.7566 | 8344.3 |
| glove | HNSW | {"M": 16, "efConstruction": 200} | {"efSearch": 64} | 0.7627 | 7169.3 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 40} | 0.7834 | 7074.7 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 48} | 0.8046 | 5996.1 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 64} | 0.8339 | 4740.7 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 80} | 0.8556 | 3974.7 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 96} | 0.8695 | 3455.0 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 128} | 0.8906 | 2340.3 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 160} | 0.9060 | 2218.0 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 192} | 0.9185 | 1919.7 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 256} | 0.9350 | 1455.2 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 320} | 0.9463 | 1155.2 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 384} | 0.9545 | 984.0 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 512} | 0.9658 | 753.0 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 768} | 0.9791 | 523.6 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 1024} | 0.9855 | 400.7 |
| glove | HNSW | {"M": 32, "efConstruction": 200} | {"efSearch": 1536} | 0.9919 | 275.4 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 1} | 0.3251 | 23023.8 |
| glove | IVF-Flat | {"nlist": 1024} | {"nprobe": 1} | 0.3977 | 25155.4 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 2} | 0.4379 | 18333.5 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 3} | 0.5039 | 17677.4 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 4} | 0.5471 | 14747.8 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 6} | 0.6035 | 11824.6 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 8} | 0.6397 | 10486.5 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 12} | 0.6877 | 8248.5 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 16} | 0.7186 | 6383.2 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 24} | 0.7581 | 4981.1 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 32} | 0.7838 | 4143.7 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 48} | 0.8188 | 2779.6 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 64} | 0.8406 | 2306.0 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 96} | 0.8705 | 1537.9 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 128} | 0.8901 | 1200.9 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 192} | 0.9156 | 861.0 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 256} | 0.9316 | 674.3 |
| glove | IVF-Flat | {"nlist": 1024} | {"nprobe": 96} | 0.9345 | 462.8 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 384} | 0.9516 | 444.8 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 512} | 0.9635 | 331.1 |
| glove | IVF-Flat | {"nlist": 1024} | {"nprobe": 192} | 0.9686 | 236.7 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 768} | 0.9782 | 237.7 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 1024} | 0.9866 | 178.3 |
| glove | IVF-Flat | {"nlist": 1024} | {"nprobe": 384} | 0.9918 | 120.0 |
| glove | IVF-Flat | {"nlist": 4096} | {"nprobe": 1536} | 0.9952 | 117.8 |
| glove | IVF-PQ | {"nlist": 1024, "m": 10} | {"nprobe": 2} | 0.2479 | 26385.6 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 1} | 0.3852 | 28049.2 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 2} | 0.5014 | 19565.4 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 3} | 0.5609 | 13841.4 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 4} | 0.5997 | 11703.8 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 6} | 0.6459 | 8330.8 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 8} | 0.6744 | 6646.9 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 12} | 0.7109 | 4641.0 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 16} | 0.7350 | 3597.3 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 24} | 0.7630 | 2484.6 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 32} | 0.7813 | 1941.8 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 16} | 0.7856 | 1274.0 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 48} | 0.8036 | 1325.7 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 64} | 0.8178 | 1015.1 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 32} | 0.8443 | 735.4 |
| glove | IVF-PQ | {"nlist": 1024, "m": 50} | {"nprobe": 128} | 0.8443 | 522.5 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 64} | 0.8920 | 388.6 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 96} | 0.9162 | 269.5 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 128} | 0.9299 | 159.7 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 192} | 0.9459 | 126.8 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 256} | 0.9550 | 74.2 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 512} | 0.9674 | 51.0 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 768} | 0.9693 | 33.1 |
| glove | IVF-PQ | {"nlist": 1024, "m": 100} | {"nprobe": 1024} | 0.9695 | 24.8 |
