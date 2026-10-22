"""Recall@k that does not count ties at the k-th neighbour as mistakes.

A returned vector is correct if its true distance to the query is no larger than the k-th smallest true
distance, not only if its id appears in the ground-truth list. When several vectors sit at exactly the k-th
distance (common for SIFT, whose vectors are integer-valued) any of them is a valid answer.
"""
import numpy as np


def true_distances(queries: np.ndarray, base: np.ndarray, ids: np.ndarray, chunk: int = 500) -> np.ndarray:
    """Exact Euclidean distance from query i to base[ids[i, j]], in float64; negative ids (missing) give inf.

    Index implementations differ in what they return as "distance" (PQ returns an approximation), so recall is
    always computed from the vectors themselves.
    """
    ids = np.asarray(ids)
    out = np.empty(ids.shape, dtype=np.float64)
    for s in range(0, len(ids), chunk):
        rows = ids[s:s + chunk]
        diff = base[np.maximum(rows, 0)].astype(np.float64) - queries[s:s + chunk, None, :].astype(np.float64)
        d = np.sqrt(np.einsum("qkd,qkd->qk", diff, diff))
        d[rows < 0] = np.inf
        out[s:s + chunk] = d
    return out


def per_query_recall_at_k(found_ids, found_dists, ref_dists, k: int = 10, rtol: float = 1e-6) -> np.ndarray:
    """Recall@k of every query, in [0, 1].

    found_ids   (nq, >=k) ids returned by the index; -1 marks a missing result.
    found_dists (nq, >=k) true distances from each query to those ids (see `true_distances`).
    ref_dists   (nq, >=k) true distances to the exact nearest neighbours, so ref_dists[:, :k] are the k smallest.

    A returned id is correct if its true distance is <= the largest of ref_dists[:, :k] (times 1 + rtol, so that
    float32 rounding in the exact search cannot turn a tie into a miss). A repeated id counts once.
    """
    found_ids, found_dists, ref_dists = (np.asarray(a) for a in (found_ids, found_dists, ref_dists))
    if min(found_ids.shape[1], found_dists.shape[1], ref_dists.shape[1]) < k:
        raise ValueError(f"need at least k={k} columns in found_ids, found_dists and ref_dists")
    found_ids, found_dists, ref_dists = found_ids[:, :k], found_dists[:, :k], ref_dists[:, :k]

    threshold = ref_dists.max(axis=1, keepdims=True) * (1.0 + rtol)
    correct = (found_ids >= 0) & (found_dists <= threshold)

    order = np.argsort(found_ids, axis=1, kind="stable")
    sorted_ids = np.take_along_axis(found_ids, order, axis=1)
    repeated_sorted = np.zeros_like(correct)
    repeated_sorted[:, 1:] = sorted_ids[:, 1:] == sorted_ids[:, :-1]
    repeated = np.empty_like(repeated_sorted)
    np.put_along_axis(repeated, order, repeated_sorted, axis=1)

    return (correct & ~repeated).sum(axis=1) / k


def recall_at_k(found_ids, found_dists, ref_dists, k: int = 10, rtol: float = 1e-6) -> float:
    """Mean recall@k over the queries; see `per_query_recall_at_k`."""
    return float(per_query_recall_at_k(found_ids, found_dists, ref_dists, k, rtol).mean())


def evaluate_recall(found_ids, queries, base, ref_dists, k: int = 10) -> float:
    """Recall@k of `found_ids` using distances recomputed from the vectors."""
    found_ids = np.asarray(found_ids)[:, :k]
    return recall_at_k(found_ids, true_distances(queries, base, found_ids), ref_dists, k)
