"""results.csv: one row for every benchmark run behind a number in the report."""
import csv
import json
from pathlib import Path

COLUMNS = [
    "part", "dataset", "experiment", "index", "mode", "n_base", "n_queries", "k", "threads",
    "build_params", "search_params", "recall_at_10",
    "qps", "ms_per_query", "seconds_median", "seconds_repeats",
    "build_seconds", "index_bytes", "bytes_per_vector", "notes",
]


def make_row(**fields) -> dict:
    """A row with every column present; dict/list values are stored as JSON."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown results column(s): {sorted(unknown)}")
    row = {c: fields.get(c, "") for c in COLUMNS}
    return {c: json.dumps(v) if isinstance(v, (dict, list)) else v for c, v in row.items()}


def replace_part(path: str | Path, part: str, rows: list[dict]) -> None:
    """Write `rows` into the csv, replacing any rows of `part` from an earlier run and keeping the others."""
    path = Path(path)
    kept = []
    if path.exists():
        with open(path, newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["part"] != part]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(kept + rows)
