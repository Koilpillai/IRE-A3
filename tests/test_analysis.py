from annbench.analysis import best_at_recall, nlist_grid, pareto_frontier, should_stop


def point(name, recall, qps):
    return {"name": name, "recall": recall, "qps": qps}


def test_nlist_grid_is_powers_of_two_around_the_square_root():
    assert nlist_grid(1_000_000) == [256, 1024, 4096]
    assert nlist_grid(1_183_514) == [256, 1024, 4096]
    assert nlist_grid(100_000) == [64, 256, 1024]


def test_sweep_stops_when_the_target_recall_is_reached():
    assert should_stop([0.5, 0.8, 0.991])
    assert not should_stop([0.5, 0.8, 0.95])


def test_sweep_stops_when_recall_stops_improving():
    assert should_stop([0.5, 0.6, 0.6005, 0.6010])
    assert not should_stop([0.5, 0.6, 0.6005, 0.65])   # one flat step is not a plateau
    assert should_stop([0.5, 0.5, 0.5])                # two flat steps in a row


def test_sweep_does_not_stop_early_on_the_first_steps():
    assert not should_stop([])
    assert not should_stop([0.3])
    assert not should_stop([0.3, 0.31])


def test_frontier_keeps_only_points_that_are_not_beaten_on_both_axes():
    points = [point("a", 0.5, 100), point("b", 0.6, 50), point("dominated", 0.55, 40), point("c", 0.9, 10)]
    assert [p["name"] for p in pareto_frontier(points)] == ["a", "b", "c"]


def test_frontier_drops_a_slower_point_with_the_same_recall():
    points = [point("slow", 0.9, 10), point("fast", 0.9, 20)]
    assert [p["name"] for p in pareto_frontier(points)] == ["fast"]


def test_frontier_can_use_other_columns():
    points = [{"tr": 0.9, "tq": 5}, {"tr": 0.8, "tq": 9}]
    assert len(pareto_frontier(points, "tr", "tq")) == 2


def test_best_at_recall_is_the_fastest_point_that_reaches_it():
    points = [point("a", 0.96, 10), point("b", 0.95, 30), point("c", 0.94, 99)]
    assert best_at_recall(points, 0.95)["name"] == "b"
    assert best_at_recall(points, 0.97) is None
