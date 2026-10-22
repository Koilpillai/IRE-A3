import numpy as np
import pytest

from annbench.recall import evaluate_recall, per_query_recall_at_k, recall_at_k, true_distances

K = 10
REF = np.arange(1.0, K + 1)[None, :]     # true distances 1..10 of one query's exact neighbours


def recall(ids, dists, ref=REF, k=K):
    return recall_at_k(np.array([ids]), np.array([dists], dtype=float), ref, k)


def test_perfect_and_zero_recall():
    assert recall(range(10), range(1, 11)) == 1.0
    assert recall(range(100, 110), range(50, 60)) == 0.0


def test_partial_recall_is_the_fraction_found():
    ids = list(range(7)) + [100, 101, 102]
    dists = list(range(1, 8)) + [20, 21, 22]
    assert recall(ids, dists) == pytest.approx(0.7)


def test_order_of_the_returned_list_does_not_matter():
    assert recall([9, 3, 0, 7, 1, 8, 2, 6, 4, 5], [10, 4, 1, 8, 2, 9, 3, 7, 5, 6]) == 1.0


def test_tie_at_the_tenth_neighbour_is_not_a_mistake():
    # Neighbours 10 and 11 are both at distance 10.0; the exact search listed id 9, the index returned id 10.
    ids = list(range(9)) + [10]
    dists = list(range(1, 10)) + [10.0]
    assert recall(ids, dists) == 1.0


def test_an_exact_tie_counts_even_with_no_tolerance():
    ids = list(range(9)) + [10]
    dists = list(range(1, 10)) + [10.0]
    assert recall_at_k(np.array([ids]), np.array([dists]), REF, K, rtol=0.0) == 1.0


def test_reference_distances_need_not_be_sorted():
    # Float32 rounding in the exact search can swap two nearly-tied neighbours around rank 10.
    ref = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 10, 9]])
    assert recall(range(10), range(1, 11), ref=ref) == 1.0


def test_just_beyond_the_tenth_neighbour_is_a_mistake():
    ids = list(range(9)) + [10]
    dists = list(range(1, 10)) + [10.001]
    assert recall(ids, dists) == pytest.approx(0.9)


def test_float_rounding_at_the_tie_is_forgiven():
    ids = list(range(9)) + [10]
    dists = list(range(1, 10)) + [10.0 * (1 + 1e-7)]
    assert recall(ids, dists) == 1.0


def test_neighbours_closer_than_the_tenth_count_even_if_not_in_the_reference_list():
    # Reference distances may come from a different (equally valid) top-10, e.g. after a tie.
    ids = list(range(10))
    dists = [0.5] + list(range(2, 11))
    assert recall(ids, dists) == 1.0


def test_missing_results_are_misses():
    ids = list(range(7)) + [-1, -1, -1]
    dists = list(range(1, 8)) + [np.inf] * 3
    assert recall(ids, dists) == pytest.approx(0.7)


def test_a_repeated_id_counts_once():
    ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dists = [1.0] * 10
    assert recall(ids, dists) == pytest.approx(0.1)


def test_only_the_first_k_results_are_used():
    ids = list(range(100, 110)) + list(range(10))
    dists = [50.0] * 10 + list(range(1, 11))
    assert recall(ids, dists) == 0.0


def test_other_values_of_k():
    ref = np.arange(1.0, 6.0)[None, :]
    assert recall([0, 1, 2, 3, 9], [1, 2, 3, 4, 99], ref=ref, k=5) == pytest.approx(0.8)


def test_per_query_values_and_their_mean():
    ids = np.array([list(range(10)), list(range(100, 110))])
    dists = np.array([np.arange(1.0, 11), np.full(10, 50.0)])
    ref = np.tile(REF, (2, 1))
    assert per_query_recall_at_k(ids, dists, ref).tolist() == [1.0, 0.0]
    assert recall_at_k(ids, dists, ref) == 0.5


def test_too_few_columns_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((1, 5), int), np.zeros((1, 5)), REF, k=10)


def test_true_distances_match_a_direct_computation():
    rng = np.random.default_rng(0)
    base, queries = rng.normal(size=(50, 8)).astype("float32"), rng.normal(size=(4, 8)).astype("float32")
    ids = rng.integers(0, 50, size=(4, 6))
    expected = np.linalg.norm(base[ids].astype(float) - queries[:, None, :].astype(float), axis=2)
    np.testing.assert_allclose(true_distances(queries, base, ids, chunk=3), expected)


def test_true_distances_of_missing_ids_are_infinite():
    base, queries = np.zeros((3, 2), "float32"), np.zeros((1, 2), "float32")
    assert true_distances(queries, base, np.array([[0, -1]])).tolist() == [[0.0, np.inf]]


def test_exact_ties_in_real_vectors_do_not_cost_recall():
    # Every vector appears 5 times, so the 10th neighbour is usually tied with several other vectors.
    rng = np.random.default_rng(1)
    base = np.repeat(rng.integers(0, 5, size=(40, 4)).astype("float32"), 5, axis=0)
    queries = rng.integers(0, 5, size=(6, 4)).astype("float32")
    dists = np.linalg.norm(base[None].astype(float) - queries[:, None].astype(float), axis=2)
    ids = np.arange(len(base))
    lowest_ids = np.array([np.lexsort((ids, d))[:K] for d in dists])    # nearest first, ties by lowest id
    highest_ids = np.array([np.lexsort((-ids, d))[:K] for d in dists])  # same distances, other tied ids
    farthest = np.argsort(-dists, axis=1)[:, :K]
    ref = true_distances(queries, base, lowest_ids)

    assert not np.array_equal(np.sort(lowest_ids, axis=1), np.sort(highest_ids, axis=1))
    assert evaluate_recall(highest_ids, queries, base, ref) == 1.0
    assert evaluate_recall(farthest, queries, base, ref) < 1.0
