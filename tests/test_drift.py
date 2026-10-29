import numpy as np
import pytest

from annbench.drift import DIM, make_drift_data
from annbench.indexes import HNSW, IVFFlat, IVFPQ
from part4_benchmark import diagnostics

SIZES = dict(n_old=3000, n_new=1500, n_queries=200)


def test_same_seed_gives_the_same_data():
    a, b = make_drift_data(2.0, **SIZES), make_drift_data(2.0, **SIZES)
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("old", "new", "queries", "direction"))


def test_different_seed_gives_different_data():
    assert not np.array_equal(make_drift_data(0.0, seed=1, **SIZES).old, make_drift_data(0.0, seed=2, **SIZES).old)


def test_shapes_and_types():
    data = make_drift_data(1.0, **SIZES)
    assert data.old.shape == (3000, DIM) and data.new.shape == (1500, DIM) and data.queries.shape == (200, DIM)
    assert data.old.dtype == data.new.dtype == data.queries.dtype == np.float32


def test_only_the_shift_changes_between_settings():
    control, moved = make_drift_data(0.0, **SIZES), make_drift_data(3.0, **SIZES)
    assert np.array_equal(control.old, moved.old)                     # the old vectors never move
    assert np.array_equal(control.direction, moved.direction)
    move = 3.0 * control.unit * control.direction
    assert np.allclose(moved.new - control.new, move, atol=1e-4)      # the new vectors and the queries only translate
    assert np.allclose(moved.queries - control.queries, move, atol=1e-4)


def test_shift_length_is_in_units_of_a_typical_distance_to_the_cluster_centre():
    data = make_drift_data(2.0, **SIZES)
    assert data.shift_length == pytest.approx(2.0 * np.sqrt(DIM))      # sigma = 1
    assert np.linalg.norm(data.direction) == pytest.approx(1.0)
    shifted, control = data.new, make_drift_data(0.0, **SIZES).new
    assert np.linalg.norm((shifted - control).mean(axis=0)) == pytest.approx(data.shift_length, rel=1e-4)


def test_old_and_new_have_the_same_distribution_in_the_control():
    data = make_drift_data(0.0, n_old=20000, n_new=20000, n_queries=200)
    assert np.abs(data.old.mean(axis=0) - data.new.mean(axis=0)).max() < 0.2
    assert np.abs(data.old.std(axis=0) - data.new.std(axis=0)).max() < 0.2


@pytest.fixture(scope="module")
def vectors():
    data = make_drift_data(0.0, n_old=4000, n_new=1000, n_queries=100)
    return data


@pytest.mark.parametrize("make", [lambda: HNSW({"M": 8, "efConstruction": 50}), lambda: IVFFlat({"nlist": 16})])
def test_copy_is_independent_and_add_extends_the_index(vectors, make):
    index = make()
    index.build(vectors.old, 1, capacity=5000) if isinstance(index, HNSW) else index.build(vectors.old, 1)
    count = (lambda i: i.index.get_current_count()) if isinstance(index, HNSW) else (lambda i: i.index.ntotal)
    other = index.copy()
    other.add(vectors.new, 1)
    assert count(index) == 4000 and count(other) == 5000
    for each in (index, other):
        each.set_search(100 if isinstance(index, HNSW) else 16)
    assert other.search(vectors.new[:1])[0][0] == 4000   # a vector is its own nearest neighbour; it exists only in the copy
    assert index.search(vectors.new[:1])[0][0] < 4000


def test_added_vectors_get_the_next_ids(vectors):
    index = IVFFlat({"nlist": 16})
    index.build(vectors.old, 1)
    index.add(vectors.new, 1)
    index.set_search(16)
    assert index.search(vectors.new[:1])[0][0] == 4000                 # a vector is its own nearest neighbour, id 4000


def test_diagnostics_scan_everything_when_every_list_is_probed(vectors):
    index = IVFFlat({"nlist": 16})
    index.build(vectors.old, 1)
    out = diagnostics(index, vectors, vectors.queries, setting=16)
    assert out["vectors_scanned_per_query"] == 4000
    assert out["mean_list_size"] == 4000 / 16 and out["max_list_size"] >= out["mean_list_size"]


def test_diagnostics_pq_error_is_larger_for_vectors_far_from_the_training_data():
    near, far = make_drift_data(0.0, n_old=4000, n_new=1000, n_queries=50), make_drift_data(4.0, n_old=4000, n_new=1000, n_queries=50)
    index = IVFPQ({"nlist": 16, "m": 16})
    index.build(near.old, 1)
    out = diagnostics(index, far, far.queries, setting=4)              # trained on near.old; far.new is shifted
    assert out["pq_squared_error_new"] > 10 * out["pq_squared_error_old"]


def test_diagnostics_hnsw_has_nothing_to_report(vectors):
    index = HNSW({"M": 8, "efConstruction": 50})
    index.build(vectors.old, 1)
    assert diagnostics(index, vectors, vectors.queries, setting=20) == {}
