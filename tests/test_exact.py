import numpy as np

from annbench.data import _to_euclidean
from annbench.exact import build_flat, faiss_threads, search
from annbench.recall import evaluate_recall, true_distances


def test_faiss_flat_matches_numpy_brute_force():
    rng = np.random.default_rng(0)
    base, queries = rng.normal(size=(500, 16)).astype("float32"), rng.normal(size=(20, 16)).astype("float32")
    ids, dists = search(build_flat(base), queries, 10)
    expected = np.linalg.norm(base[None].astype(float) - queries[:, None].astype(float), axis=2)
    np.testing.assert_array_equal(ids, np.argsort(expected, axis=1)[:, :10])
    np.testing.assert_allclose(dists, np.sort(expected, axis=1)[:, :10], rtol=1e-4)
    assert evaluate_recall(ids, queries, base, true_distances(queries, base, ids)) == 1.0


def test_one_query_at_a_time_gives_the_same_neighbours_as_a_batch():
    rng = np.random.default_rng(1)
    base, queries = rng.normal(size=(300, 8)).astype("float32"), rng.normal(size=(5, 8)).astype("float32")
    index = build_flat(base)
    batch_ids, _ = search(index, queries, 5)
    with faiss_threads(1):
        single_ids = np.vstack([search(index, queries[i:i + 1], 5)[0] for i in range(len(queries))])
    np.testing.assert_array_equal(batch_ids, single_ids)


def test_faiss_threads_restores_the_previous_setting():
    import faiss
    before = faiss.omp_get_max_threads()
    with faiss_threads(1):
        assert faiss.omp_get_max_threads() == 1
    assert faiss.omp_get_max_threads() == before


def test_angular_distances_become_euclidean_distances_of_unit_vectors():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(2, 7))
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cosine_distance = 1.0 - a @ b
    np.testing.assert_allclose(_to_euclidean(np.array([cosine_distance]), "angular"), [np.linalg.norm(a - b)])
    np.testing.assert_allclose(_to_euclidean(np.array([3.5]), "euclidean"), [3.5])
