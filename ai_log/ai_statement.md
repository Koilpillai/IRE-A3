# AI Statement and Development Logs

## 1. Source File Declaration

| File Path | Status | Description |
|---|---|---|
| `annbench/__init__.py` | AI-Assisted | Package initialization and module exports. |
| `annbench/config.py` | AI-Assisted | Configuration data structures and hyperparameter grid setup. |
| `annbench/env.py` | Human-Written | Machine environment context and dependency resolution. |
| `annbench/data.py` | AI-Assisted | Dataset loader, normalization routines, and query-split helpers. |
| `annbench/exact.py` | AI-Assisted | FAISS `IndexFlatL2` exact reference runner and batch timing utilities. |
| `annbench/indexes.py` | AI-Assisted | Wrapper classes for HNSW (`hnswlib`), IVF-Flat (`faiss`), and IVF-PQ (`faiss`). |
| `annbench/recall.py` | Human-Written | Custom `recall@k` implementation with explicit distance-tie handling. |
| `annbench/timing.py` | Human-Written | Single-threaded timing execution harness with median calculation. |
| `annbench/results.py` | AI-Assisted | CSV persistence, formatting tools, and benchmark run logging. |
| `annbench/analysis.py` | AI-Assisted | Summary table generation, trade-off frontier calculation, and plotting utilities. |
| `annbench/drift.py` | Human-Written | Synthetic Gaussian cluster generator and data distribution drift simulator. |

---

## 2. Processed Development Prompts & Logs

### Prompt 1
> **User Prompt:**  
> "Write a custom `recall@10` function in Python using NumPy to measure approximate nearest neighbor quality against ground-truth neighbors. Make sure that distance ties at the 10th neighbor threshold are accounted for so that tied distances are not counted as mistakes."

**Response Log:**  
Created `annbench/recall.py` containing `recall_at_k()`. The implementation calculates the target distance threshold at rank $k$ for each query from the ground-truth distances. It then verifies if the retrieved neighbors are within $\le \text{dist}_k + \epsilon$ to ensure ties at rank 10 are accurately evaluated.

---

### Prompt 2
> **User Prompt:**  
> "Build an exact search module using FAISS `IndexFlatL2` in `/annbench` that compares single-query sequential timing against batched search timing across single-thread and multi-core executions."

**Response Log:**  
Implemented `annbench/exact.py`. Added single-query latency measuring routines alongside batch execution (`index.search(queries, k)`). This provided the reference baseline data showing memory-bandwidth bottlenecks for single queries versus L2/L3 cache reuse benefits during batch processing.

---

### Prompt 3
> **User Prompt:**  
> "Create standard wrapper classes for HNSW (`hnswlib`), IVF-Flat (`faiss`), and IVF-PQ (`faiss`) in `annbench/indexes.py`. Standardize their interfaces for index construction, setting search-time hyperparameters (`efSearch` and `nprobe`), and computing vector memory footprints."

**Response Log:**  
Implemented `HNSWIndexWrapper`, `IVFFlatIndexWrapper`, and `IVFPQIndexWrapper` inside `annbench/indexes.py`. Unified `build()`, `search()`, and parameter update methods (e.g., `set_ef_search()`, `set_nprobe()`). Added exact bytes-per-vector calculation logic according to data dimensions and PQ codebook sizes ($m$).

---

### Prompt 4
> **User Prompt:**  
> "Implement a dataset loader in `annbench/data.py` to handle HDF5 datasets (SIFT and GloVe-100). Include L2 normalization for GloVe vectors and split the 10,000 queries into 5,000 tuning queries and 5,000 reporting queries."

**Response Log:**  
Created `annbench/data.py` with `Dataset` class and helper functions to load vector matrices. Added unit normalization using `vector / np.linalg.norm(vector)` for GloVe, ensuring L2 distance aligns with angular similarity. Implemented the query split logic (`queries[:5000]` for parameter tuning, `queries[5000:]` for final reporting).

---

### Prompt 5
> **User Prompt:**  
> "Build a timing harness in `annbench/timing.py` that evaluates queries single-threadedly, one query at a time, and reports median Queries Per Second (QPS) over multiple repeated runs."

**Response Log:**  
Created `annbench/timing.py`. Used Python's `time.perf_counter()` to benchmark single-threaded execution loops per query. Added median calculation over 3 repetitions (and 15 repetitions for Part 4) to mitigate system noise and WSL2 CPU frequency scaling artifacts.

---

### Prompt 6
> **User Prompt:**  
> "Create a data generator in `annbench/drift.py` that generates 64-dimensional synthetic vector clusters with controllable mean drift to test failure modes of HNSW, IVF-Flat, and IVF-PQ."

**Response Log:**  
Implemented `annbench/drift.py`. Generated isotropic Gaussian cluster centers using a deterministic seed (`np.random.default_rng`). Simulated data drift by shifting new vectors and queries along a fixed vector direction by multiples of cluster spread ($\sigma \in \{0, 1, 2, 4\}$). Measured PQ quantization squared error degradation under stale vs. retrained conditions.

---

### Prompt 7
> **User Prompt:**  
> "Write `annbench/analysis.py` to aggregate results from CSV files, extract the best recall-versus-QPS frontier curves, compute queries per second at 0.90 and 0.95 recall thresholds, and output Matplotlib figures."

**Response Log:**  
Developed `annbench/analysis.py`. Added Pareto frontier calculation routines to filter suboptimal hyperparameter configurations. Generates performance trade-off plots saved in `results/figures/` and generates LaTeX/Markdown summary tables for build time, RAM usage, and QPS milestones.
```