"""Exact (brute-force) search with FAISS: the reference for everything else."""
import contextlib

import faiss
import numpy as np


@contextlib.contextmanager
def faiss_threads(n: int):
    """Run FAISS with `n` OpenMP threads inside the block, then restore the previous setting."""
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(n)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


def build_flat(base: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    return index


def search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, Euclidean distances); FAISS itself reports squared L2."""
    sq_dists, ids = index.search(queries, k)
    return ids, np.sqrt(np.maximum(sq_dists, 0.0))
