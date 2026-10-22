"""Loading the ann-benchmarks HDF5 files (SIFT and GloVe-100)."""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import GT_K, N_TUNE

# `stored`: the metric the file's `distances` dataset is expressed in.
DATASETS = {
    "sift": {"file": "sift-128-euclidean.hdf5", "normalise": False, "stored": "euclidean"},
    "glove": {"file": "glove-100-angular.hdf5", "normalise": True, "stored": "angular"},
}


@dataclass
class Dataset:
    name: str
    base: np.ndarray                # (n, d) float32, the vectors to index
    queries: np.ndarray             # (nq, d) float32
    tune: slice                     # rows of `queries` to tune on
    report: slice                   # rows of `queries` to report final numbers on
    file_ids: np.ndarray | None     # (nq, 100) neighbours stored in the file; None if `base` is a subset
    file_dists: np.ndarray | None   # their distances, converted to Euclidean on the vectors as loaded

    @property
    def subset(self) -> bool:
        return self.file_ids is None


def _normalise_rows(x: np.ndarray) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("cannot normalise a zero vector")
    x /= norms


def _to_euclidean(stored: np.ndarray, metric: str) -> np.ndarray:
    """Convert the file's distances to Euclidean distance between (normalised) vectors.

    The angular file stores cosine distance 1 - cos(a, b); for unit vectors |a - b|^2 = 2 - 2 cos(a, b).
    """
    stored = stored.astype(np.float64)
    return stored if metric == "euclidean" else np.sqrt(2.0 * stored)


def load_dataset(name: str, data_dir: str | Path, n_base: int | None = None,
                 n_queries_per_half: int = N_TUNE) -> Dataset:
    """Load `name`, normalising GloVe (vectors and queries) so that Euclidean distance is used throughout.

    With `n_base` the first `n_base` vectors are indexed; the neighbours stored in the file refer to the full
    set, so they are dropped. With `n_queries_per_half` < 5000 only the first that many queries of each half
    are kept (the tune half stays first, the report half second).
    """
    spec = DATASETS[name]
    with h5py.File(Path(data_dir) / spec["file"], "r") as f:
        if f.attrs["distance"] != spec["stored"]:
            raise ValueError(f"{spec['file']} is {f.attrs['distance']!r}, expected {spec['stored']!r}")
        full_size = f["train"].shape[0]
        base = np.ascontiguousarray(f["train"][:n_base], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        file_ids = f["neighbors"][:].astype(np.int64)
        file_dists = _to_euclidean(f["distances"][:], spec["stored"])
    assert file_ids.shape[1] == GT_K

    if spec["normalise"]:
        _normalise_rows(base)
        _normalise_rows(queries)

    n = n_queries_per_half
    if n < N_TUNE:
        keep = np.r_[0:n, N_TUNE:N_TUNE + n]
        queries, file_ids, file_dists = queries[keep], file_ids[keep], file_dists[keep]
    tune, report = slice(0, n), slice(n, 2 * n)

    if len(base) < full_size:
        file_ids = file_dists = None
    return Dataset(name, base, queries, tune, report, file_ids, file_dists)
