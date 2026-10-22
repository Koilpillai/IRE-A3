"""Part 1: exact baseline.

For each dataset: build exact FAISS search, check it against the neighbours stored in the data file, and time it
one query at a time and with all queries in one batch. Writes the `part1` rows of results.csv.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import faiss
import numpy as np

from annbench import env, results
from annbench.config import FULL, GT_K, K, QUICK, REPEATS
from annbench.data import DATASETS, Dataset, load_dataset
from annbench.exact import build_flat, faiss_threads, search
from annbench.recall import evaluate_recall, recall_at_k, true_distances
from annbench.timing import summarise, time_repeats

ROOT = Path(__file__).resolve().parent
MIN_AGREEMENT = 0.999   # recall of our exact search against the file (either way round) below this is an error


def check_against_file(ds: Dataset, ids: np.ndarray, dists: np.ndarray) -> dict:
    """Compare our exact top-100 (`ids`, with true distances `dists`) with the neighbours stored in the file."""
    file_dists = true_distances(ds.queries, ds.base, ds.file_ids)
    same_set = (np.sort(ids[:, :K], axis=1) == np.sort(ds.file_ids[:, :K], axis=1)).all(axis=1)
    diff = np.abs(np.sort(dists, axis=1) - ds.file_dists)
    id_overlap = (ids[:, :K, None] == ds.file_ids[:, None, :K]).any(axis=2).mean()  # recall that ignores ties
    return {
        "n_queries": len(ds.queries),
        "recall_ours_vs_file": recall_at_k(ids, dists, file_dists, K),
        "recall_file_vs_ours": recall_at_k(ds.file_ids, file_dists, dists, K),
        "naive_id_overlap_at_10": float(id_overlap),
        "queries_with_identical_top10_sets": int(same_set.sum()),
        "queries_with_different_top10_sets": int((~same_set).sum()),
        "max_abs_distance_diff_top100": float(diff.max()),
        "max_rel_distance_diff_top100": float((diff / np.maximum(ds.file_dists, 1e-12)).max()),
    }


def time_exact(index: faiss.IndexFlatL2, ds: Dataset, ref_dists: np.ndarray, repeats: int,
               threads: int) -> list[dict]:
    """Time exact top-10 search over the report queries: one at a time (1 thread), and as one batch (1 and `threads`)."""
    queries = ds.queries[ds.report]
    rows = [queries[i:i + 1] for i in range(len(queries))]  # sliced up front so that only searching is timed

    def one_at_a_time():
        return np.vstack([index.search(r, K)[1] for r in rows])

    def batch():
        return index.search(queries, K)[1]

    plans = [
        ("one_at_a_time", 1, one_at_a_time, lambda: [index.search(r, K) for r in rows[:50]]),
        ("batch", 1, batch, lambda: index.search(queries[:200], K)),
        ("batch", threads, batch, lambda: index.search(queries[:200], K)),
    ]
    timings = []
    for mode, n_threads, run, warmup in plans:
        with faiss_threads(n_threads):
            seconds, found = time_repeats(run, repeats, warmup)
        recall = evaluate_recall(found, queries, ds.base, ref_dists[ds.report], K)
        timings.append({"mode": mode, "threads": n_threads, "seconds_repeats": seconds, "recall_at_10": recall,
                        **summarise(seconds, len(queries))})
    return timings


def run_dataset(name: str, data_dir: Path, profile: dict, repeats: int, threads: int) -> list[dict]:
    ds = load_dataset(name, data_dir, **profile)
    n, d = ds.base.shape
    print(f"\n=== {name}: {n:,} vectors x {d}, {len(ds.queries):,} queries "
          f"(report on the {ds.report.stop - ds.report.start:,} in the second half)", flush=True)

    start = time.perf_counter()
    index = build_flat(ds.base)
    build = time.perf_counter() - start
    index_bytes = index.ntotal * d * 4   # IndexFlatL2 stores the raw float32 vectors, nothing else
    common = dict(part="part1", dataset=name, index="Flat", n_base=n, k=K, build_seconds=build,
                  index_bytes=index_bytes, bytes_per_vector=index_bytes / n)

    ids, _ = search(index, ds.queries, GT_K)              # reference for everything below (and later parts)
    ref_dists = true_distances(ds.queries, ds.base, ids)  # exact distances of those neighbours, float64
    rows = []

    if ds.subset:
        print("  agreement with the file: skipped (the stored neighbours refer to the full dataset)")
    else:
        agreement = check_against_file(ds, ids, ref_dists)
        print("  agreement with the neighbours stored in the file:")
        print(f"    recall@10, ours vs file: {agreement['recall_ours_vs_file']:.5f}   "
              f"file vs ours: {agreement['recall_file_vs_ours']:.5f}")
        print(f"    same ids without tie handling: {agreement['naive_id_overlap_at_10']:.5f}; "
              f"identical top-10 id sets: {agreement['queries_with_identical_top10_sets']:,} of "
              f"{agreement['n_queries']:,} queries")
        print(f"    top-100 distances: max abs diff {agreement['max_abs_distance_diff_top100']:.2e}, "
              f"max rel diff {agreement['max_rel_distance_diff_top100']:.2e}")
        if min(agreement["recall_ours_vs_file"], agreement["recall_file_vs_ours"]) < MIN_AGREEMENT:
            raise RuntimeError(f"{name}: exact search disagrees with the stored neighbours: {agreement}")
        rows.append(results.make_row(
            **common, experiment="agreement_with_file", mode="batch", threads=faiss.omp_get_max_threads(),
            n_queries=agreement["n_queries"], recall_at_10=agreement["recall_ours_vs_file"], notes=agreement))

    timings = time_exact(index, ds, ref_dists, repeats, threads)
    for t in timings:
        rows.append(results.make_row(**common, experiment="exact_timing", n_queries=ds.report.stop - ds.report.start,
                                     **t))
    print_timings(timings)
    del ds, index
    gc.collect()
    return rows


def print_timings(timings: list[dict]) -> None:
    for t in timings:
        print(f"  {t['mode']:>13s}, {t['threads']:2d} thread(s): {t['qps']:9.1f} QPS  {t['ms_per_query']:8.3f} ms/query"
              f"   recall@10 {t['recall_at_10']:.4f}   repeats (s): {', '.join(f'{s:.2f}' for s in t['seconds_repeats'])}")
    single, batch_1, batch_all = (t["qps"] for t in timings)
    print(f"  batching on one thread is {batch_1 / single:.1f}x faster than one query at a time; "
          f"using more threads adds another {batch_all / batch_1:.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--quick", action="store_true", help="first 100,000 vectors and 1,000 queries per half")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: results (results_quick with --quick)")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--threads", type=int, default=env.physical_cores(),
                        help="threads for the multi-threaded batch timing (default: physical cores)")
    args = parser.parse_args()

    profile = QUICK if args.quick else FULL
    results_dir = args.results_dir or ROOT / ("results_quick" if args.quick else "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "environment.json").write_text(json.dumps(env.collect(), indent=2) + "\n")

    rows = []
    for name in args.datasets:
        rows += run_dataset(name, args.data_dir, profile, args.repeats, args.threads)
    results.replace_part(results_dir / "results.csv", "part1", rows)
    print(f"\nwrote {len(rows)} rows to {results_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
