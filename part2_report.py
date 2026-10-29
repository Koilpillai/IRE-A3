"""Part 2 figures and tables, made from the part1 and part2 rows of results.csv.

Selection uses only the tune half: for each index, the points of the best trade-off curve (the Pareto frontier of
recall and speed over all build settings and search parameters) and the setting that is fastest at recall 0.90 / 0.95
are chosen on the first 5,000 queries. The numbers shown for them are the ones measured on the report half.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from annbench import results  # noqa: E402
from annbench.analysis import best_at_recall, pareto_frontier  # noqa: E402

ROOT = Path(__file__).resolve().parent
INDEXES = ["HNSW", "IVF-Flat", "IVF-PQ"]
TARGETS = [0.90, 0.95]
NAMES = {"sift": ("SIFT", 128), "glove": ("GloVe-100", 100)}

# Reference palette, light mode, first three categorical slots (validated all-pairs); grey for the exact baseline.
STYLE = {
    "HNSW": {"color": "#2a78d6", "marker": "o"},
    "IVF-Flat": {"color": "#eb6834", "marker": "s"},
    "IVF-PQ": {"color": "#1baf7a", "marker": "^"},
    "Exact (Flat)": {"color": "#52514e", "marker": "D"},
}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"


def sweep_points(rows: list[dict], dataset: str, index: str) -> list[dict]:
    """One point per (build setting, search parameter), with its tune-half and report-half measurements."""
    by_key = {}
    for r in rows:
        if r["part"] == "part2" and r["dataset"] == dataset and r["index"] == index \
                and r["experiment"] in ("sweep_tune", "sweep_report"):
            by_key.setdefault((r["build_params"], r["search_params"]), {})[r["experiment"]] = r
    points = []
    for (build, search), pair in by_key.items():
        tune, report = pair["sweep_tune"], pair["sweep_report"]
        points.append({
            "build": json.loads(build), "search": json.loads(search),
            "tune_recall": float(tune["recall_at_10"]), "tune_qps": float(tune["qps"]),
            "recall": float(report["recall_at_10"]), "qps": float(report["qps"]),
            "build_seconds": float(report["build_seconds"]), "index_bytes": int(float(report["index_bytes"])),
            "bytes_per_vector": float(report["bytes_per_vector"]),
        })
    return points


def exact_baseline(rows: list[dict], dataset: str) -> dict:
    """Part 1's exact search, one query at a time on one thread."""
    for r in rows:
        if r["part"] == "part1" and r["dataset"] == dataset and r["experiment"] == "exact_timing" \
                and r["mode"] == "one_at_a_time":
            return {"qps": float(r["qps"]), "recall": float(r["recall_at_10"]), "build_seconds": float(r["build_seconds"]),
                    "index_bytes": int(float(r["index_bytes"])), "bytes_per_vector": float(r["bytes_per_vector"])}
    raise SystemExit(f"no part1 exact timing for {dataset} in results.csv - run part1_exact.py first")


def describe(p: dict) -> str:
    build = ", ".join(f"{k}={v}" for k, v in p["build"].items() if k != "efConstruction")
    search = ", ".join(f"{k}={v}" for k, v in p["search"].items())
    return f"{build}, {search}"


def plot(dataset: str, n_base: int, frontiers: dict, exact: dict, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.set_yscale("log")
    for target in TARGETS:
        ax.axvline(target, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(target, 1.01, f"{target:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=INK_MUTED)
    ax.axhline(exact["qps"], color=STYLE["Exact (Flat)"]["color"], lw=1, ls=(0, (1, 2)), zorder=1)
    for index, frontier in frontiers.items():
        if not frontier:
            continue
        pts = sorted(frontier, key=lambda p: p["recall"])
        s = STYLE[index]
        ax.plot([p["recall"] for p in pts], [p["qps"] for p in pts], color=s["color"], lw=2, marker=s["marker"],
                ms=6, mec=SURFACE, mew=1, label=index, zorder=3)
    s = STYLE["Exact (Flat)"]
    ax.plot([exact["recall"]], [exact["qps"]], ls="", marker=s["marker"], ms=8, color=s["color"], mec=SURFACE,
            mew=1, label="Exact (Flat)", zorder=4)
    lowest = min(p["recall"] for f in frontiers.values() for p in f)
    ax.set_xlim(max(0.0, int(lowest * 10) / 10), 1.01)
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (one query at a time, 1 thread)")
    name, dim = NAMES[dataset]
    ax.set_title(f"{name}: {n_base:,} vectors, {dim}-d", loc="left", fontsize=10)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.1), fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"part2_recall_vs_qps_{dataset}.{ext}", dpi=200)
    plt.close(fig)


def markdown(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def write_table(out_dir: Path, name: str, header: list[str], body: list[list[str]], note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    (out_dir / f"{name}.md").write_text(markdown(header, body) + (f"\n{note}\n" if note else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="read results_quick/ instead of results/")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or ROOT / ("results_quick" if args.quick else "results")
    rows = results.read_rows(results_dir / "results.csv")
    datasets = sorted({r["dataset"] for r in rows if r["part"] == "part2"}, key=["sift", "glove"].index)

    speed_body, cost_body, ceiling_body, frontier_body = [], [], [], []
    for dataset in datasets:
        exact = exact_baseline(rows, dataset)
        frontiers = {}
        speed_body.append([dataset, "Exact (Flat)"] + [f"{exact['qps']:.1f}", f"{exact['recall']:.3f}"] * 2 + ["-"])
        cost_body.append([dataset, "Exact (Flat)", "-", f"{exact['build_seconds']:.1f}",
                          f"{exact['index_bytes'] / 2**20:.0f}", f"{exact['bytes_per_vector']:.0f}"])
        for index in INDEXES:
            points = sweep_points(rows, dataset, index)
            if not points:
                continue
            frontier = pareto_frontier(points, "tune_recall", "tune_qps")
            frontiers[index] = frontier
            for p in frontier:
                frontier_body.append([dataset, index, json.dumps(p["build"]), json.dumps(p["search"]),
                                      f"{p['recall']:.4f}", f"{p['qps']:.1f}"])

            chosen = {t: best_at_recall(points, t, "tune_recall", "tune_qps") for t in TARGETS}
            row = [dataset, index]
            for t in TARGETS:
                p = chosen[t]
                row += [f"{p['qps']:.1f}", f"{p['recall']:.3f}"] if p else ["not reached", "-"]
            best_tune = max(points, key=lambda p: p["tune_recall"])
            row.append(describe(chosen[0.95]) if chosen[0.95] else f"- (best: {describe(best_tune)})")
            speed_body.append(row)

            p = chosen[0.95] or best_tune
            flag = "" if chosen[0.95] else " *"
            cost_body.append([dataset, index, json.dumps(p["build"]).replace('"', ""), f"{p['build_seconds']:.1f}{flag}",
                              f"{p['index_bytes'] / 2**20:.0f}{flag}", f"{p['bytes_per_vector']:.0f}{flag}"])

        for index in INDEXES:
            for build in sorted({json.dumps(p["build"]) for p in sweep_points(rows, dataset, index)}):
                pts = [p for p in sweep_points(rows, dataset, index) if json.dumps(p["build"]) == build]
                top = max(pts, key=lambda p: p["recall"])
                check = [r for r in rows if r["part"] == "part2" and r["dataset"] == dataset and r["index"] == index
                         and r["experiment"] == "ceiling_check" and r["build_params"] == build]
                ceiling_body.append([dataset, index, build.replace('"', ""), f"{top['recall']:.3f}",
                                     json.dumps(top["search"]).replace('"', ""),
                                     f"{float(check[0]['recall_at_10']):.3f}" if check else "-"])
        n_base = int(next(r["n_base"] for r in rows if r["part"] == "part2" and r["dataset"] == dataset))
        plot(dataset, n_base, frontiers, exact, results_dir / "figures")

    tables = results_dir / "tables"
    write_table(tables, "part2_qps_at_recall",
                ["dataset", "index", "QPS @ recall 0.90", "recall reached", "QPS @ recall 0.95", "recall reached",
                 "setting at 0.95"], speed_body,
                "Settings are chosen on the tune queries; QPS and recall are measured on the report queries "
                "(median of 3), so the recall reached can fall slightly below the target and the plotted curves are not "
                "perfectly monotone.")
    write_table(tables, "part2_build_cost",
                ["dataset", "index", "build setting at 0.95", "build time (s)", "index size (MiB)", "bytes per vector"],
                cost_body, "* recall 0.95 was not reached; the setting with the highest recall is shown.")
    write_table(tables, "part2_recall_ceilings",
                ["dataset", "index", "build setting", "highest recall in sweep", "at", "recall with nprobe=nlist (500 queries)"],
                ceiling_body)
    write_table(tables, "part2_frontier", ["dataset", "index", "build", "search", "recall@10", "QPS"], frontier_body)
    print((tables / "part2_qps_at_recall.md").read_text())
    print((tables / "part2_build_cost.md").read_text())
    print((tables / "part2_recall_ceilings.md").read_text())
    print(f"figures in {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
