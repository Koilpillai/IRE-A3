"""Part 3 (compression): IVF-PQ with and without exact re-ranking of the top 100 candidates, on SIFT.

Hypothesis: part3_hypothesis.md (committed before this was run).

For each number of PQ sub-vectors m one IVF-PQ index is built; the plain version and the version that re-ranks the
top 100 PQ candidates against the original vectors then search the same codes. For each, nprobe is swept exactly as
in Part 2 (one query at a time, one thread, tune half timed once, report half timed `--repeats` times). Extra rows:
  ceiling_check     recall with every list scanned (nprobe = nlist), on 500 tune queries;
  candidate_recall  the fraction of the true top 10 that is inside the top 100 PQ candidates, on 1,000 tune queries
                    (not timed): the most that re-ranking could recover.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from annbench import env, results
from annbench.analysis import nlist_grid
from annbench.config import FULL, K, QUICK, REPEATS
from annbench.data import load_dataset
from annbench.exact import build_flat, faiss_threads, search
from annbench.indexes import IVFPQ, RefinedIVFPQ
from annbench.recall import candidate_recall_at_k, true_distances
from part2_benchmark import CEILING_QUERIES, measure, sweep

ROOT = Path(__file__).resolve().parent
M_VALUES = (16, 32, 64, 128)   # PQ sub-vectors, i.e. bytes per code: 8, 4, 2 and 1 dimensions of SIFT each
CANDIDATES = 100               # how many PQ results are re-ranked
MAX_NPROBE = 256               # the sweep never goes beyond this; nprobe = nlist is checked separately (ceiling)
CANDIDATE_QUERIES = 1000


def candidate_rows(plain: IVFPQ, ds, ref_dists: np.ndarray, grid: list[int], threads: int, common: dict) -> list[dict]:
    """For each nprobe, how much of the true top 10 is in the top `CANDIDATES` results of the PQ codes alone."""
    queries = ds.queries[ds.tune][:CANDIDATE_QUERIES]
    ref = ref_dists[ds.tune][:CANDIDATE_QUERIES]
    rows = []
    for nprobe in sorted(set(grid) | {plain.params["nlist"]}):
        plain.set_search(nprobe)
        with faiss_threads(threads):
            ids = plain.index.search(queries, CANDIDATES)[1].astype(np.int64)
        value = candidate_recall_at_k(ids, true_distances(queries, ds.base, ids), ref, K)
        meaning = f"recall_at_10 is the fraction of the true top 10 inside the top {CANDIDATES} PQ results"
        rows.append(results.make_row(**dict(common, mode="batch", threads=threads, notes={"meaning": meaning}),
                                     experiment="candidate_recall", n_queries=len(queries),
                                     search_params={"nprobe": nprobe}, recall_at_10=value))
        print(f"    nprobe={nprobe:<5d} true top-10 inside the PQ top {CANDIDATES}: {value:.4f}", flush=True)
    return rows


def run_setting(m: int, nlist: int, ds, ref_dists: np.ndarray, args, common: dict) -> list[dict]:
    plain = IVFPQ({"nlist": nlist, "m": m})
    build = plain.build(ds.base, args.threads)
    refined = RefinedIVFPQ(plain, ds.base, CANDIDATES)
    grid = [p for p in plain.search_grid() if p <= MAX_NPROBE]
    rows = []
    for index in (plain, refined):
        size = index.size_bytes()
        row_common = dict(common, index=index.kind, build_params=index.params, build_seconds=build["build_seconds"],
                          index_bytes=size, bytes_per_vector=size / len(ds.base), notes=build)
        print(f"  {index.kind} {json.dumps(index.params)}: PQ built in {build['build_seconds']:.1f}s, "
              f"{size / len(ds.base):.0f} bytes/vector", flush=True)
        rows += sweep(index, ds, ref_dists, args.repeats, row_common, grid)

        index.set_search(nlist)   # scanning every list leaves only the error of the compression (and of the re-ranking)
        tune_q = ds.queries[ds.tune][:CEILING_QUERIES]
        got = measure(index, tune_q, ds.base, ref_dists[ds.tune][:CEILING_QUERIES], repeats=1)
        rows.append(results.make_row(**row_common, experiment="ceiling_check", n_queries=CEILING_QUERIES,
                                     search_params={"nprobe": nlist}, **got))
        print(f"    ceiling (nprobe=nlist): recall {got['recall_at_10']:.4f}", flush=True)

        if index is plain:   # the candidates are the same for both versions, so this is measured once
            rows += candidate_rows(plain, ds, ref_dists, grid, args.threads, row_common)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="first 100,000 vectors and 1,000 queries per half")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: results (results_quick with --quick)")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--threads", type=int, default=env.physical_cores(),
                        help="threads for building indexes and for the exact reference (default: physical cores)")
    parser.add_argument("--resume", action="store_true", help="keep finished values of m from an earlier run")
    args = parser.parse_args()

    profile = QUICK if args.quick else FULL
    results_dir = args.results_dir or ROOT / ("results_quick" if args.quick else "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "environment.json").write_text(json.dumps(env.collect(), indent=2) + "\n")
    results_path = results_dir / "results.csv"
    if not args.resume:
        results.replace_part(results_path, "part3", [])
    # rows of a value of m are saved together, so any re-ranked row means that m is finished
    done = {r["build_params"] for r in results.read_rows(results_path)
            if r["part"] == "part3" and r["index"] == RefinedIVFPQ.kind}

    ds = load_dataset("sift", args.data_dir, **profile)
    n, d = ds.base.shape
    nlist = nlist_grid(n)[1]   # the same nlist as the IVF-PQ settings of Part 2
    print(f"=== sift: {n:,} vectors x {d}, nlist {nlist}", flush=True)

    exact = build_flat(ds.base)
    with faiss_threads(args.threads):
        ids, _ = search(exact, ds.queries, K)   # reference neighbours for recall
    ref_dists = true_distances(ds.queries, ds.base, ids)
    del exact

    common = dict(part="part3", dataset="sift", mode="one_at_a_time", n_base=n, k=K, threads=1)
    for m in M_VALUES:
        params = {"nlist": nlist, "m": m, "candidates": CANDIDATES}
        if json.dumps(params) in done:
            print(f"  m={m}: already done, skipping")
            continue
        results.append_rows(results_path, run_setting(m, nlist, ds, ref_dists, args, common))
        gc.collect()
    print(f"\nresults in {results_path}")


if __name__ == "__main__":
    main()
