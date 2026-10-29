"""Choosing build settings and picking points off recall/speed sweeps (no FAISS or hnswlib in here)."""
import math


def nlist_grid(n: int) -> list[int]:
    """Three IVF list counts around sqrt(n): a quarter of it, it, and four times it, as powers of two."""
    root = math.sqrt(n)
    return [2 ** round(math.log2(root * f)) for f in (0.25, 1.0, 4.0)]


def should_stop(recalls: list[float], target: float = 0.99, min_gain: float = 0.002, patience: int = 2) -> bool:
    """Whether a sweep (recall after each step of increasing search effort) can stop.

    It stops once recall reaches `target`, or once `patience` consecutive steps each gained less than `min_gain`
    (the index has hit its ceiling, so more effort only costs time).
    """
    if recalls and recalls[-1] >= target:
        return True
    if len(recalls) <= patience:
        return False
    tail = recalls[-(patience + 1):]
    return all(b - a < min_gain for a, b in zip(tail, tail[1:]))


def pareto_frontier(points: list[dict], recall: str = "recall", speed: str = "qps") -> list[dict]:
    """The points that no other point beats on both recall and speed, ordered by increasing recall."""
    frontier, fastest_so_far = [], -math.inf
    for p in sorted(points, key=lambda p: (-p[recall], -p[speed])):
        if p[speed] > fastest_so_far:
            frontier.append(p)
            fastest_so_far = p[speed]
    return frontier[::-1]


def best_at_recall(points: list[dict], target: float, recall: str = "recall", speed: str = "qps") -> dict | None:
    """The fastest point with recall >= target, or None if no point reaches it."""
    reaching = [p for p in points if p[recall] >= target]
    return max(reaching, key=lambda p: p[speed]) if reaching else None
