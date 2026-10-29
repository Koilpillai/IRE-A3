"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ on SIFT and GloVe-100.

For every build setting of every index, sweep the search-time parameter and measure recall@10 and speed
(one query at a time, one thread) on both query halves:
  tune   (first half)  timed once; used only to choose which points go in the report (part2_report.py);
  report (second half) timed `--repeats` times, the median is reported.
Rows are saved per build setting as soon as it is finished, so an interrupted run can continue with --resume.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from annbench import env, results
from annbench.analysis import should_stop
from annbench.config import FULL, K, QUICK, REPEATS
from annbench.data import DATASETS, Dataset, load_dataset
from annbench.exact import build_flat, faiss_threads, search
from annbench.indexes import KINDS, build_settings
from annbench.recall import evaluate_recall, true_distances
from annbench.timing import summarise, time_repeats

ROOT = Path(__file__).resolve().parent
CEILING_QUERIES = 500   # queries used to measure the recall ceiling (search effort at its maximum)


def measure(index, queries: np.ndarray, base: np.ndarray, ref_dists: np.ndarray, repeats: int) -> dict:
    """Recall@10 and speed of answering `queries` one at a time on one thread."""
    rows = [queries[i:i + 1] for i in range(len(queries))]   # sliced up front so that only searching is timed

    def run():
        return [index.search(r) for r in rows]

    gc.collect()
    gc.disable()
    try:
        with faiss_threads(1):
            seconds, out = time_repeats(run, repeats, warmup=lambda: [index.search(r) for r in rows[:100]])
    finally:
        gc.enable()
    found = np.vstack(out).astype(np.int64)
    return {"recall_at_10": evaluate_recall(found, queries, base, ref_dists, K), "seconds_repeats": seconds,
            **summarise(seconds, len(queries))}


def sweep(index, ds: Dataset, ref_dists: np.ndarray, repeats: int, common: dict, grid: list[int]) -> list[dict]:
    """Rows for `index` at each search-parameter value of `grid` in turn, stopping early once recall levels off.

    The tune half is timed once and the report half `repeats` times; both give recall@10 and speed.
    """
    rows = []
    tune_q, report_q = ds.queries[ds.tune], ds.queries[ds.report]
    recalls = []
    for value in grid:
        index.set_search(value)
        tune = measure(index, tune_q, ds.base, ref_dists[ds.tune], repeats=1)
        report = measure(index, report_q, ds.base, ref_dists[ds.report], repeats=repeats)
        for experiment, m, queries in (("sweep_tune", tune, tune_q), ("sweep_report", report, report_q)):
            rows.append(results.make_row(**common, experiment=experiment, n_queries=len(queries),
                                         search_params={index.search_param: value}, **m))
        print(f"    {index.search_param}={value:<5d} tune: recall {tune['recall_at_10']:.4f} {tune['qps']:8.1f} QPS"
              f"   report: recall {report['recall_at_10']:.4f} {report['qps']:8.1f} QPS", flush=True)
        recalls.append(tune["recall_at_10"])
        if should_stop(recalls):
            break
    return rows


def benchmark_config(index, ds: Dataset, ref_dists: np.ndarray, repeats: int, threads: int, common: dict) -> list[dict]:
    """Build `index`, sweep its search parameter, and return the results rows."""
    build = index.build(ds.base, threads)
    size = index.size_bytes()
    common = dict(common, build_params=index.params, build_seconds=build["build_seconds"], index_bytes=size,
                  bytes_per_vector=size / len(ds.base), notes=build)
    print(f"  {index.kind} {json.dumps(index.params)}: built in {build['build_seconds']:.1f}s, "
          f"{size / len(ds.base):.0f} bytes/vector", flush=True)

    rows = sweep(index, ds, ref_dists, repeats, common, index.search_grid())
    tune_q = ds.queries[ds.tune]

    if index.kind == "IVF-PQ":   # scanning every list leaves only the error of the compression itself
        index.set_search(index.params["nlist"])
        m = measure(index, tune_q[:CEILING_QUERIES], ds.base, ref_dists[ds.tune][:CEILING_QUERIES], repeats=1)
        rows.append(results.make_row(**common, experiment="ceiling_check", n_queries=CEILING_QUERIES,
                                     search_params={index.search_param: index.params["nlist"]}, **m))
        print(f"    ceiling (nprobe=nlist): recall {m['recall_at_10']:.4f}", flush=True)
    return rows


def run_dataset(name: str, args, profile: dict, results_path: Path, done: set) -> None:
    ds = load_dataset(name, args.data_dir, **profile)
    n, d = ds.base.shape
    print(f"\n=== {name}: {n:,} vectors x {d}", flush=True)

    exact = build_flat(ds.base)
    with faiss_threads(args.threads):
        ids, _ = search(exact, ds.queries, K)   # reference neighbours for recall
    ref_dists = true_distances(ds.queries, ds.base, ids)
    del exact

    common = dict(part="part2", dataset=name, mode="one_at_a_time", n_base=n, k=K, threads=1)
    for kind in args.indexes:
        for params in build_settings(kind, n, d):
            index = KINDS[kind](params)
            if (name, index.kind, json.dumps(params)) in done:
                print(f"  {index.kind} {json.dumps(params)}: already done, skipping")
                continue
            rows = benchmark_config(index, ds, ref_dists, args.repeats, args.threads, dict(common, index=index.kind))
            results.append_rows(results_path, rows)
            del index
            gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--indexes", nargs="+", choices=list(KINDS), default=list(KINDS))
    parser.add_argument("--quick", action="store_true", help="first 100,000 vectors and 1,000 queries per half")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: results (results_quick with --quick)")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--threads", type=int, default=env.physical_cores(),
                        help="threads for building indexes and for the exact reference (default: physical cores)")
    parser.add_argument("--resume", action="store_true", help="keep finished build settings from an earlier run")
    args = parser.parse_args()

    profile = QUICK if args.quick else FULL
    results_dir = args.results_dir or ROOT / ("results_quick" if args.quick else "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "environment.json").write_text(json.dumps(env.collect(), indent=2) + "\n")
    results_path = results_dir / "results.csv"

    if not args.resume:
        results.replace_part(results_path, "part2", [])
    done = {(r["dataset"], r["index"], r["build_params"]) for r in results.read_rows(results_path)
            if r["part"] == "part2"}
    for name in args.datasets:
        run_dataset(name, args, profile, results_path, done)
    print(f"\nresults in {results_path}")


if __name__ == "__main__":
    main()
