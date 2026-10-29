"""Part 4 (drifting data): build on old vectors, add shifted vectors, query shifted queries, then retrain / rebuild.

Hypothesis: part4_hypothesis.md (committed before this was run). Data: annbench/drift.py (seeded generator).

For every shift (0 = the control) and every index two versions are measured on the same data and queries:
  stale      built (and, for IVF, trained) on the old vectors only, then the new vectors are added to it;
  retrained  built from scratch on old + new (IVF: new centroids and codebooks; HNSW: a new graph).
The search setting of each index (nprobe or efSearch) is chosen once, on the tune half of the control queries, as the
smallest one that reaches recall 0.95, and then kept fixed for every shift and both versions. Recall and speed (one query
at a time, one thread, median of `--repeats`, 15 by default) are measured on the report half. Extra numbers explain what happened:
the number of vectors the IVF lists make a query scan, and the error of the PQ codes on the new vectors.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from annbench import env, results
from annbench.analysis import nlist_grid
from annbench.config import K
from annbench.drift import make_drift_data
from annbench.exact import build_flat, faiss_threads, search
from annbench.indexes import EF_CONSTRUCTION, HNSW, IVFFlat, IVFPQ
from annbench.recall import true_distances
from part2_benchmark import measure

ROOT = Path(__file__).resolve().parent
SHIFTS = (0.0, 1.0, 2.0, 4.0)   # in units of a typical vector's distance to its cluster centre; 0 is the control
TARGET = 0.95                   # the search setting is the smallest one that reaches this recall in the control
PQ_M = 64                       # 1 dimension per code byte: the only m (of 16, 32, 64) that reaches 0.95 in the control
FULL = {"n_old": 200_000, "n_new": 100_000, "n_queries": 2000}
QUICK = {"n_old": 20_000, "n_new": 10_000, "n_queries": 2000}
DATASET = "synthetic_drift"
DIAGNOSTIC_VECTORS = 2000       # new vectors used to measure the error of the PQ codes
REPEATS = 15                    # each timed pass takes only about 0.1 s here, so 3 repeats left a 5-35% spread (first run)


def make_indexes(n_old: int) -> dict:
    """The three indexes with the one fixed build setting each (the same as Parts 2 and 3 where one exists)."""
    nlist = nlist_grid(n_old)[1]
    return {"HNSW": HNSW({"M": 16, "efConstruction": EF_CONSTRUCTION}), "IVF-Flat": IVFFlat({"nlist": nlist}),
            "IVF-PQ": IVFPQ({"nlist": nlist, "m": PQ_M})}


def fresh(index):
    """A new, unbuilt index with the same build setting."""
    return type(index)(index.params)


def diagnostics(index, data, queries: np.ndarray, setting: int) -> dict:
    """Numbers that help to explain a result; empty for HNSW."""
    out = {}
    if index.kind in ("IVF-Flat", "IVF-PQ"):
        ivf = index.index
        sizes = np.array([ivf.invlists.list_size(i) for i in range(ivf.nlist)])
        probes = ivf.quantizer.search(queries, setting)[1]
        out.update(mean_list_size=float(sizes.mean()), max_list_size=int(sizes.max()),
                   vectors_scanned_per_query=float(sizes[probes].sum(axis=1).mean()))
    if index.kind == "IVF-PQ":
        for name, x in (("old", data.old[:DIAGNOSTIC_VECTORS]), ("new", data.new[:DIAGNOSTIC_VECTORS])):
            error = ((x - ivf.sa_decode(ivf.sa_encode(x))) ** 2).sum(axis=1).mean()
            out[f"pq_squared_error_{name}"] = float(error)
    return out


def choose_setting(index, tune_q, base, ref_dists, common: dict) -> tuple[int, list[dict]]:
    """Smallest value of the search parameter whose recall on the tune queries reaches TARGET, and the rows tried."""
    rows, best = [], None
    for value in index.search_grid():
        index.set_search(value)
        got = measure(index, tune_q, base, ref_dists, repeats=1)
        rows.append(results.make_row(**common, experiment="choose_setting", n_queries=len(tune_q),
                                     search_params={index.search_param: value}, **got))
        print(f"    {index.search_param}={value:<5d} tune recall {got['recall_at_10']:.4f}", flush=True)
        best = value
        if got["recall_at_10"] >= TARGET:
            return value, rows
    raise SystemExit(f"{index.kind} never reaches recall {TARGET} in the control (best at {index.search_param}={best}); "
                     "the experiment's set-up needs another setting")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="20,000 old and 10,000 new vectors instead of 200,000 and 100,000")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: results (results_quick with --quick)")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--threads", type=int, default=env.physical_cores(),
                        help="threads for building and adding to indexes and for the exact reference (default: physical cores)")
    args = parser.parse_args()

    profile = QUICK if args.quick else FULL
    results_dir = args.results_dir or ROOT / ("results_quick" if args.quick else "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "environment.json").write_text(json.dumps(env.collect(), indent=2) + "\n")
    results_path = results_dir / "results.csv"
    results.replace_part(results_path, "part4", [])

    assert SHIFTS[0] == 0.0, "the control must come first: it chooses the search settings"
    n_old, n_new, nq = profile["n_old"], profile["n_new"], profile["n_queries"]
    n_tune = nq // 2
    total = n_old + n_new
    old = make_drift_data(0.0, **profile).old
    print(f"=== synthetic drift: {n_old:,} old + {n_new:,} new vectors, {nq} queries, shifts {SHIFTS}", flush=True)

    built, old_build = {}, {}
    for kind, index in make_indexes(n_old).items():
        stats = index.build(old, args.threads, capacity=total) if kind == "HNSW" else index.build(old, args.threads)
        built[kind], old_build[kind] = index, stats["build_seconds"]
        print(f"  {kind} {json.dumps(index.params)}: built on the old vectors in {stats['build_seconds']:.1f}s", flush=True)

    settings = {}
    for shift in SHIFTS:
        data = make_drift_data(shift, **profile)
        base = np.vstack([data.old, data.new])
        with faiss_threads(args.threads):
            ids, _ = search(build_flat(base), data.queries, K)   # reference neighbours for recall
        ref_dists = true_distances(data.queries, base, ids)
        tune_q, report_q = data.queries[:n_tune], data.queries[n_tune:]
        print(f"\n-- shift {shift:g} ({data.shift_length:.1f} in vector length; one unit is {data.unit:.1f})", flush=True)

        rows = []
        for kind, old_index in built.items():
            stale = old_index.copy()
            add = stale.add(data.new, args.threads)
            retrained = fresh(old_index)
            full = retrained.build(base, args.threads)
            arms = {"stale": (stale, add["add_seconds"], {"old_build_seconds": old_build[kind], **add}),
                    "retrained": (retrained, full["build_seconds"], full)}
            if kind not in settings:   # the control: choose the search setting on the tune half, with the stale index
                print(f"  {kind}: choosing the search setting", flush=True)
                common = dict(part="part4", dataset=DATASET, mode="one_at_a_time", n_base=total, k=K, threads=1,
                              index=kind, build_params=old_index.params,
                              notes={"shift": shift, "arm": "stale", "half": "tune"})
                settings[kind], tried = choose_setting(stale, tune_q, base, ref_dists[:n_tune], common)
                rows += tried
            value = settings[kind]
            for arm, (index, seconds, stats) in arms.items():
                index.set_search(value)
                size = index.size_bytes()
                got = measure(index, report_q, base, ref_dists[n_tune:], args.repeats)
                extra = diagnostics(index, data, report_q, value)
                rows.append(results.make_row(
                    part="part4", dataset=DATASET, experiment=f"drift_{arm}", index=kind, mode="one_at_a_time",
                    n_base=total, n_queries=len(report_q), k=K, threads=1, build_params=old_index.params,
                    search_params={index.search_param: value}, build_seconds=seconds, index_bytes=size,
                    bytes_per_vector=size / total, notes={"shift": shift, "arm": arm, "n_old": n_old, "n_new": n_new, **stats, **extra}, **got))
                print(f"  {kind:8s} {arm:9s} {index.search_param}={value:<4d} recall {got['recall_at_10']:.4f} "
                      f"{got['qps']:9.1f} QPS  ({seconds:.1f}s to {'add' if arm == 'stale' else 'rebuild'})  {extra}",
                      flush=True)
            del stale, retrained
            gc.collect()
        results.append_rows(results_path, rows)
    print(f"\nresults in {results_path}")


if __name__ == "__main__":
    main()
