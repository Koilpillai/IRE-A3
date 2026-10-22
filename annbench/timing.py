"""Wall-clock timing helpers."""
import statistics
import time


def time_repeats(run, repeats: int = 3, warmup=None):
    """Call `run()` `repeats` times; return (seconds of each repeat, result of the last repeat).

    `warmup`, if given, is called once first and not timed (to fault in memory and warm the caches).
    """
    if warmup is not None:
        warmup()
    seconds, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = run()
        seconds.append(time.perf_counter() - start)
    return seconds, result


def summarise(seconds: list[float], n_queries: int) -> dict:
    """Median-based speed of a timing: seconds, queries per second and milliseconds per query."""
    median = statistics.median(seconds)
    return {"seconds_median": median, "qps": n_queries / median, "ms_per_query": 1e3 * median / n_queries}
