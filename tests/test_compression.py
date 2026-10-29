import numpy as np
import pytest

from annbench.config import K
from annbench.exact import build_flat, search
from annbench.indexes import IVFPQ, RefinedIVFPQ
from annbench.recall import candidate_recall_at_k, evaluate_recall, true_distances


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(20, 16)) * 4
    base = (centres[rng.integers(0, 20, 3000)] + rng.normal(size=(3000, 16))).astype("float32")
    queries = (centres[rng.integers(0, 20, 100)] + rng.normal(size=(100, 16))).astype("float32")
    ids, _ = search(build_flat(base), queries, K)
    return base, queries, true_distances(queries, base, ids)


@pytest.fixture(scope="module")
def indexes(data):
    base, _, _ = data
    plain = IVFPQ({"nlist": 16, "m": 4})   # 4 dimensions per code byte: coarse on purpose
    plain.build(base, threads=1)
    return plain, RefinedIVFPQ(plain, base, candidates=100)


def recall_of(index, base, queries, ref):
    found = np.vstack([index.search(queries[i:i + 1]) for i in range(len(queries))]).astype(np.int64)
    return evaluate_recall(found, queries, base, ref, K)


def test_reranking_improves_recall_of_coarse_codes(data, indexes):
    base, queries, ref = data
    plain, refined = indexes
    refined.set_search(16)
    assert recall_of(plain, base, queries, ref) < recall_of(refined, base, queries, ref)


def test_scanning_every_list_with_all_points_as_candidates_is_exact(data, indexes):
    base, queries, ref = data
    plain, _ = indexes
    refined = RefinedIVFPQ(plain, base, candidates=len(base))   # re-rank everything the lists return
    refined.set_search(16)
    assert recall_of(refined, base, queries, ref) == 1.0


def test_both_versions_share_nprobe(indexes):
    plain, refined = indexes
    refined.set_search(3)
    assert plain.index.nprobe == 3
    plain.set_search(5)
    assert refined.plain.index.nprobe == 5


def test_size_includes_the_original_vectors(data, indexes):
    base, _, _ = data
    plain, refined = indexes
    assert refined.size_bytes() - plain.size_bytes() >= base.nbytes
    assert refined.size_bytes() - plain.size_bytes() < base.nbytes + 1000   # and little else


def test_params_record_the_candidates(indexes):
    _, refined = indexes
    assert refined.params == {"nlist": 16, "m": 4, "candidates": 100}


def test_candidate_recall_counts_neighbours_anywhere_in_the_list():
    ref = np.arange(1.0, 13.0)[None, :]               # true distances 1..12; the 10th neighbour is at 10
    ids = np.arange(20)[None, :]
    dists = np.r_[np.full(15, 50.0), np.arange(1.0, 6.0)][None, :]   # only 5 candidates are near enough, all late
    assert candidate_recall_at_k(ids, dists, ref, k=10) == 0.5


def test_candidate_recall_is_capped_at_one_and_ignores_repeats():
    ref = np.ones((1, 10))
    assert candidate_recall_at_k(np.arange(30)[None, :], np.ones((1, 30)), ref, k=10) == 1.0   # ties: 30 qualify
    assert candidate_recall_at_k(np.zeros((1, 30), dtype=int), np.ones((1, 30)), ref, k=10) == 0.1   # one id, repeated
    assert candidate_recall_at_k(np.full((1, 30), -1), np.zeros((1, 30)), ref, k=10) == 0.0   # missing results
