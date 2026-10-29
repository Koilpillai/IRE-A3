import numpy as np
import pytest

from annbench.config import K
from annbench.exact import build_flat, search
from annbench.indexes import HNSW, IVFFlat, IVFPQ, PQ_M, build_settings
from annbench.recall import evaluate_recall, true_distances


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(20, 16)) * 4
    base = (centres[rng.integers(0, 20, 3000)] + rng.normal(size=(3000, 16))).astype("float32")
    queries = (centres[rng.integers(0, 20, 100)] + rng.normal(size=(100, 16))).astype("float32")
    ids, _ = search(build_flat(base), queries, K)
    return base, queries, true_distances(queries, base, ids)


def recall_of(index, base, queries, ref):
    found = np.vstack([index.search(queries[i:i + 1]) for i in range(len(queries))]).astype(np.int64)
    return evaluate_recall(found, queries, base, ref, K)


def test_ivf_flat_scanning_every_list_is_exact(data):
    base, queries, ref = data
    index = IVFFlat({"nlist": 16})
    assert index.build(base, threads=1)["build_seconds"] > 0
    index.set_search(16)
    assert recall_of(index, base, queries, ref) == 1.0


def test_ivf_flat_recall_grows_with_nprobe(data):
    base, queries, ref = data
    index = IVFFlat({"nlist": 16})
    index.build(base, threads=1)
    recalls = []
    for nprobe in (1, 4, 16):
        index.set_search(nprobe)
        recalls.append(recall_of(index, base, queries, ref))
    assert recalls == sorted(recalls) and recalls[0] < recalls[-1]


def test_ivf_pq_is_smaller_than_ivf_flat_and_loses_recall(data):
    base, queries, ref = data
    flat, pq = IVFFlat({"nlist": 16}), IVFPQ({"nlist": 16, "m": 4})
    flat.build(base, threads=1)
    pq.build(base, threads=1)
    assert pq.size_bytes() < flat.size_bytes() / 3
    flat.set_search(16)
    pq.set_search(16)
    assert recall_of(pq, base, queries, ref) < recall_of(flat, base, queries, ref)


def test_search_returns_k_ids_for_one_query(data):
    base, queries, _ = data
    index = IVFFlat({"nlist": 16})
    index.build(base, threads=1)
    index.set_search(4)
    assert index.search(queries[:1]).shape == (1, K)


def test_hnsw_reaches_high_recall(data):
    pytest.importorskip("hnswlib")
    base, queries, ref = data
    index = HNSW({"M": 8, "efConstruction": 100})
    assert index.build(base, threads=1)["build_seconds"] > 0
    index.set_search(200)
    assert recall_of(index, base, queries, ref) > 0.95
    assert index.size_bytes() > base.nbytes


def test_search_grids_start_low_and_respect_nlist():
    assert HNSW({}).search_grid()[0] == K
    grid = IVFFlat({"nlist": 100}).search_grid()
    assert grid[0] == 1 and max(grid) <= 100


@pytest.mark.parametrize("d", sorted(PQ_M))
def test_build_settings_are_valid(d):
    for kind, expected in (("hnsw", 3), ("ivfflat", 3), ("ivfpq", 5)):
        assert len(build_settings(kind, 1_000_000, d)) == expected
    assert all(d % p["m"] == 0 for p in build_settings("ivfpq", 1_000_000, d))
