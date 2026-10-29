"""The three index types behind one interface: HNSW (hnswlib), IVF-Flat and IVF-PQ (FAISS).

Every index is built from the base vectors, has one search-time parameter that trades recall for speed,
and answers one query at a time.
"""
import os
import tempfile
import time

import faiss
import numpy as np

from .analysis import nlist_grid
from .config import K
from .exact import faiss_threads

EF_CONSTRUCTION = 200
PQ_M = {128: (16, 32, 64, 128), 100: (10, 20, 50, 100)}   # PQ sub-vectors m per dimension d: 8, 4, 2 and 1 (SIFT) or 10, 5, 2 and 1 (GloVe) dims each
EF_GRID = [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
NPROBE_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]


class HNSW:
    kind = "HNSW"
    search_param = "efSearch"

    def __init__(self, params: dict):
        self.params = params

    def build(self, base: np.ndarray, threads: int) -> dict:
        import hnswlib
        n, d = base.shape
        self.index = hnswlib.Index(space="l2", dim=d)
        self.index.init_index(max_elements=n, ef_construction=self.params["efConstruction"],
                              M=self.params["M"], random_seed=100)
        start = time.perf_counter()
        self.index.add_items(base, np.arange(n), num_threads=threads)
        self.index.set_num_threads(1)
        return {"build_seconds": time.perf_counter() - start}

    def search_grid(self) -> list[int]:
        return EF_GRID

    def set_search(self, value: int) -> None:
        self.index.set_ef(value)

    def search(self, row: np.ndarray) -> np.ndarray:
        return self.index.knn_query(row, k=K)[0]

    def size_bytes(self) -> int:
        """Size of the saved index file (vectors, graph and ids)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hnsw.bin")
            self.index.save_index(path)
            return os.path.getsize(path)


class _IVF:
    search_param = "nprobe"

    def __init__(self, params: dict):
        self.params = params

    def _make(self, d: int) -> faiss.IndexIVF:
        raise NotImplementedError

    def build(self, base: np.ndarray, threads: int) -> dict:
        self.quantizer = faiss.IndexFlatL2(base.shape[1])   # must stay alive as long as the index does
        self.index = self._make(base.shape[1])
        with faiss_threads(threads):
            start = time.perf_counter()
            self.index.train(base)
            train = time.perf_counter() - start
            self.index.add(base)
        total = time.perf_counter() - start
        return {"build_seconds": total, "train_seconds": train, "add_seconds": total - train}

    def search_grid(self) -> list[int]:
        return [p for p in NPROBE_GRID if p <= self.params["nlist"]]

    def set_search(self, value: int) -> None:
        self.index.nprobe = value

    def search(self, row: np.ndarray) -> np.ndarray:
        return self.index.search(row, K)[1]

    def size_bytes(self) -> int:
        """Size of the serialised index (vectors or codes, ids, centroids)."""
        return int(faiss.serialize_index(self.index).nbytes)


class IVFFlat(_IVF):
    kind = "IVF-Flat"

    def _make(self, d: int) -> faiss.IndexIVFFlat:
        return faiss.IndexIVFFlat(self.quantizer, d, self.params["nlist"])


class IVFPQ(_IVF):
    kind = "IVF-PQ"

    def _make(self, d: int) -> faiss.IndexIVFPQ:
        return faiss.IndexIVFPQ(self.quantizer, d, self.params["nlist"], self.params["m"], 8)


KINDS = {"hnsw": HNSW, "ivfflat": IVFFlat, "ivfpq": IVFPQ}


def build_settings(kind: str, n: int, d: int) -> list[dict]:
    """The build settings tried for an index type on n vectors of dimension d."""
    nlists = nlist_grid(n)
    if kind == "hnsw":
        return [{"M": m, "efConstruction": EF_CONSTRUCTION} for m in (8, 16, 32)]
    if kind == "ivfflat":
        return [{"nlist": nlist} for nlist in nlists]
    if kind == "ivfpq":
        ms = PQ_M[d]
        return [{"nlist": nlists[1], "m": m} for m in ms] + [{"nlist": nlists[2], "m": ms[1]}]
    raise ValueError(kind)
