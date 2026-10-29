"""Part 3 figure and tables, made from the part1, part2 and part3 rows of results.csv.

As in Part 2, the setting that reaches a recall target is chosen on the tune half; the numbers shown for it are the
ones measured on the report half. The curves show every nprobe of the sweep (report half), so nothing is selected there.
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from annbench import results  # noqa: E402
from annbench.analysis import best_at_recall  # noqa: E402
from part2_report import GRID, INK, INK_MUTED, SURFACE, write_table  # noqa: E402

ROOT = Path(__file__).resolve().parent
PLAIN, RERANK = "IVF-PQ", "IVF-PQ+rerank"
TARGETS = [0.90, 0.95]
COMPARE_NPROBE = (4, 16, 32)
# Ordinal blue ramp (steps 250, 350, 500, 650 of the reference palette, validated with --ordinal): lighter = fewer
# sub-vectors. The two versions are told apart by line style and marker, not by colour.
M_COLOURS = {16: "#86b6ef", 32: "#5598e7", 64: "#256abf", 128: "#104281"}
VARIANTS = {PLAIN: {"ls": "-", "marker": "o", "label": "plain"},
            RERANK: {"ls": (0, (5, 2)), "marker": "s", "label": "re-ranked (top 100)"}}


def point(tune: dict, report: dict) -> dict:
    """One sweep point: the tune-half and report-half rows of the same build setting and search value."""
    search = json.loads(report["search_params"])
    return {
        "build": json.loads(report["build_params"]), "search": search, "nprobe": search["nprobe"],
        "tune_recall": float(tune["recall_at_10"]), "tune_qps": float(tune["qps"]),
        "recall": float(report["recall_at_10"]), "qps": float(report["qps"]),
        "build_seconds": float(report["build_seconds"]), "index_bytes": int(float(report["index_bytes"])),
        "bytes_per_vector": float(report["bytes_per_vector"]),
    }


def load_points(rows: list[dict]) -> dict:
    """{(index kind, m): list of points sorted by nprobe}, each with tune and report recall and speed."""
    by_key = {}
    for r in rows:
        if r["part"] == "part3" and r["experiment"] in ("sweep_tune", "sweep_report"):
            by_key.setdefault((r["index"], r["build_params"], r["search_params"]), {})[r["experiment"]] = r
    out = {}
    for (index, build, _), pair in by_key.items():
        out.setdefault((index, json.loads(build)["m"]), []).append(point(pair["sweep_tune"], pair["sweep_report"]))
    return {k: sorted(v, key=lambda p: p["nprobe"]) for k, v in out.items()}


def side_rows(rows: list[dict], experiment: str) -> dict:
    """{(index kind, m): {nprobe: value}} of the recall in rows of `experiment` (ceiling_check, candidate_recall)."""
    out = {}
    for r in rows:
        if r["part"] == "part3" and r["experiment"] == experiment:
            build, search = json.loads(r["build_params"]), json.loads(r["search_params"])
            out.setdefault((r["index"], build["m"]), {})[search["nprobe"]] = float(r["recall_at_10"])
    return out


def exact_baseline(rows: list[dict]) -> dict:
    """Part 1's exact search, one query at a time on one thread."""
    for r in rows:
        if r["part"] == "part1" and r["dataset"] == "sift" and r["experiment"] == "exact_timing" \
                and r["mode"] == "one_at_a_time":
            return {"qps": float(r["qps"]), "bytes_per_vector": float(r["bytes_per_vector"])}
    raise SystemExit("no part1 exact timing for sift in results.csv - run part1_exact.py first")


def ivf_flat_points(rows: list[dict], nlist: int) -> list[dict]:
    """Part 2's IVF-Flat sweep with the same nlist, in the same form as `load_points` gives (empty if not run)."""
    by_search = {}
    for r in rows:
        if r["part"] == "part2" and r["dataset"] == "sift" and r["index"] == "IVF-Flat" \
                and r["experiment"] in ("sweep_tune", "sweep_report") and json.loads(r["build_params"]) == {"nlist": nlist}:
            by_search.setdefault(r["search_params"], {})[r["experiment"]] = r
    pts = [point(pair["sweep_tune"], pair["sweep_report"]) for pair in by_search.values()]
    return sorted(pts, key=lambda p: p["nprobe"])


def plot(points: dict, exact: dict, flat: list[dict], n_base: int, nlist: int, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) recall against speed: one curve per m and version, one point per nprobe
    left.set_yscale("log")
    for target in TARGETS:
        left.axvline(target, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    left.axhline(exact["qps"], color=INK_MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    left.text(0.31, exact["qps"] * 1.1, "exact search", fontsize=8, color=INK_MUTED)
    for (index, m), pts in sorted(points.items(), key=lambda kv: (kv[0][1], kv[0][0] != PLAIN)):
        s = VARIANTS[index]
        left.plot([p["recall"] for p in pts], [p["qps"] for p in pts], color=M_COLOURS[m], ls=s["ls"], lw=2,
                  marker=s["marker"], ms=5, mec=SURFACE, mew=0.8, zorder=3)
    left.set_xlim(0.3, 1.0)
    left.set_xlabel("recall@10")
    left.set_ylabel("queries per second (one at a time, 1 thread)")
    left.set_title("(a) Recall against speed (nprobe from 1 up)", loc="left", fontsize=10)

    # (b) memory against the highest recall reached
    for m in sorted({m for _, m in points}):
        best = {index: max(points[(index, m)], key=lambda p: p["recall"]) for index in (PLAIN, RERANK)}
        right.plot([best[PLAIN]["bytes_per_vector"], best[RERANK]["bytes_per_vector"]],
                   [best[PLAIN]["recall"], best[RERANK]["recall"]], color=M_COLOURS[m], lw=1.2, zorder=2)
        for index, p in best.items():
            right.plot([p["bytes_per_vector"]], [p["recall"]], ls="", marker=VARIANTS[index]["marker"], ms=8,
                       color=M_COLOURS[m], mec=SURFACE, mew=1, zorder=3)
        right.annotate(f"m={m}", (best[PLAIN]["bytes_per_vector"], best[PLAIN]["recall"]), xytext=(6, -4),
                       textcoords="offset points", ha="left", va="top", fontsize=8, color=INK_MUTED)
    arrow = {"arrowstyle": "-", "color": INK_MUTED, "lw": 0.7}
    right.plot([exact["bytes_per_vector"]], [1.0], ls="", marker="D", ms=7, color=INK_MUTED, mec=SURFACE, zorder=4)
    right.annotate("exact (Flat)", (exact["bytes_per_vector"], 1.0), xytext=(exact["bytes_per_vector"] - 40, 1.04),
                   ha="right", va="center", fontsize=8, color=INK_MUTED, arrowprops=arrow)
    if flat:
        at = (flat[0]["bytes_per_vector"], max(p["recall"] for p in flat))
        right.plot([at[0]], [at[1]], ls="", marker="P", ms=8, color=INK_MUTED, mec=SURFACE, zorder=4)
        right.annotate(f"IVF-Flat, nlist={nlist} (Part 2)", at, xytext=(at[0] + 10, 0.78), ha="center", va="center",
                       fontsize=8, color=INK_MUTED, arrowprops=arrow)
    top = max(points[(RERANK, m)][-1]["bytes_per_vector"] for m in {m for _, m in points})
    right.annotate("re-ranked: PQ codes + original vectors", (top, 0.99), xytext=(top, 0.66), ha="right", va="center",
                   fontsize=8, color=INK_MUTED, arrowprops=arrow)
    right.set_xlabel("bytes per vector (index size / number of vectors)")
    right.set_ylabel("highest recall@10 reached in the sweep")
    right.set_ylim(0.55, 1.07)
    right.set_xlim(0, top * 1.08)
    right.set_title("(b) Memory against the recall reached", loc="left", fontsize=10)

    for ax in (left, right):
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=M_COLOURS[m], lw=3, label=f"m={m} ({128 // m} dim{'s' if m < 128 else ''}/byte)")
               for m in sorted(M_COLOURS)]
    handles += [Line2D([], [], color=INK_MUTED, ls=s["ls"], marker=s["marker"], ms=5, lw=1.5, label=s["label"])
                for s in VARIANTS.values()]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"SIFT, {n_base:,} vectors, IVF-PQ with nlist={nlist}: plain against re-ranking the top 100", x=0.01,
                 ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"part3_compression.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="read results_quick/ instead of results/")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or ROOT / ("results_quick" if args.quick else "results")
    rows = results.read_rows(results_dir / "results.csv")

    points = load_points(rows)
    if not points:
        raise SystemExit("no part3 rows in results.csv - run part3_benchmark.py first")
    ceilings, candidates = side_rows(rows, "ceiling_check"), side_rows(rows, "candidate_recall")
    ms = sorted({m for _, m in points})
    first = next(r for r in rows if r["part"] == "part3")
    n_base = int(first["n_base"])
    nlist = json.loads(first["build_params"])["nlist"]
    exact = exact_baseline(rows)

    flat = ivf_flat_points(rows, nlist)

    def summary_row(m: str, version: str, pts: list[dict], ceiling: str) -> list[str]:
        row = [m, version, f"{pts[0]['bytes_per_vector']:.0f}", f"{pts[0]['index_bytes'] / 2**20:.0f}",
               f"{pts[0]['build_seconds']:.1f}", f"{max(p['recall'] for p in pts):.3f}", ceiling]
        for t in TARGETS:
            p = best_at_recall(pts, t, "tune_recall", "tune_qps")
            row += [f"{p['qps']:.0f}", f"{p['recall']:.3f}", str(p["nprobe"])] if p else ["not reached", "-", "-"]
        return row

    summary = []
    for m in ms:
        for index in (PLAIN, RERANK):
            summary.append(summary_row(str(m), VARIANTS[index]["label"], points[(index, m)],
                                       f"{ceilings[(index, m)][nlist]:.3f}"))
    if flat:
        summary.append(summary_row("-", f"IVF-Flat, nlist={nlist} (Part 2, for reference)", flat, "-"))

    overhead, ratios = [], []
    for m in ms:
        plain = {p["nprobe"]: p for p in points[(PLAIN, m)]}
        rerank = {p["nprobe"]: p for p in points[(RERANK, m)]}
        shared = sorted(set(plain) & set(rerank))
        ratio = [rerank[n]["qps"] / plain[n]["qps"] for n in shared]
        extra_us = [1e6 * (1 / rerank[n]["qps"] - 1 / plain[n]["qps"]) for n in shared]
        ratios.append([str(m), f"{statistics.median(ratio):.2f}", f"{min(ratio):.2f}", f"{max(ratio):.2f}",
                       f"{statistics.median(extra_us):.0f}", f"{min(extra_us):.0f}", f"{max(extra_us):.0f}", str(len(ratio))])
        for n in COMPARE_NPROBE:
            if n in plain and n in rerank:
                overhead.append([str(m), str(n), f"{plain[n]['recall']:.3f}", f"{rerank[n]['recall']:.3f}",
                                 f"{plain[n]['qps']:.0f}", f"{rerank[n]['qps']:.0f}",
                                 f"{rerank[n]['qps'] / plain[n]['qps']:.2f}", f"{candidates[(PLAIN, m)][n]:.3f}"])

    tables = results_dir / "tables"
    write_table(tables, "part3_summary",
                ["m", "version", "bytes per vector", "index size (MiB)", "PQ build time (s)", "highest recall (nprobe <= 256)",
                 "recall at nprobe=nlist", "QPS @ 0.90", "recall reached", "nprobe", "QPS @ 0.95", "recall reached", "nprobe"],
                summary, "nprobe is chosen on the tune queries; QPS and recall are measured on the report queries (median "
                "of 3). 'recall at nprobe=nlist' uses 500 tune queries. Build time is that of the shared PQ index; "
                "wrapping it for re-ranking does not copy the vectors and was not timed.")
    write_table(tables, "part3_same_nprobe",
                ["m", "nprobe", "recall plain", "recall re-ranked", "QPS plain", "QPS re-ranked", "QPS ratio (re-ranked / plain)",
                 f"true top 10 inside the PQ top {100}"], overhead,
                "Report queries, one at a time on one thread; the last column is measured on 1,000 tune queries.")
    write_table(tables, "part3_qps_ratio", ["m", "median QPS ratio", "lowest", "highest", "median extra time per query (us)", "lowest", "highest",
                 "nprobe values compared"], ratios,
                "QPS of the re-ranked version divided by QPS of the plain one at the same nprobe, over every nprobe of the "
                "sweep; the extra time is the difference of the two times per query.")
    plot(points, exact, flat, n_base, nlist, results_dir / "figures")
    for name in ("part3_summary", "part3_same_nprobe", "part3_qps_ratio"):
        print((tables / f"{name}.md").read_text())
    print(f"figure in {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
