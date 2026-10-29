"""Part 4 figure and tables, made from the part4 rows of results.csv.

Every number is on the report half of the queries, at the one search setting per index that was chosen on the tune half
of the control queries (shift 0), so no setting is chosen on the numbers shown.
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
from part2_report import GRID, INK, INK_MUTED, STYLE, SURFACE, write_table  # noqa: E402

ROOT = Path(__file__).resolve().parent
INDEXES = ["HNSW", "IVF-Flat", "IVF-PQ"]
ARMS = {"stale": {"ls": "-", "marker": "o", "label": "stale (old build + new vectors added)"},
        "retrained": {"ls": (0, (5, 2)), "marker": "s", "label": "retrained / rebuilt on all vectors"}}


def load(rows: list[dict]) -> dict:
    """{(index, shift, arm): measurement}, with the notes (diagnostics, build times) unpacked."""
    out = {}
    for r in rows:
        if r["part"] == "part4" and r["experiment"] in ("drift_stale", "drift_retrained"):
            notes = json.loads(r["notes"])
            out[(r["index"], notes["shift"], notes["arm"])] = {
                "recall": float(r["recall_at_10"]), "qps": float(r["qps"]), "seconds": float(r["build_seconds"]),
                "repeats": json.loads(r["seconds_repeats"]),
                "setting": json.loads(r["search_params"]), "n_queries": int(r["n_queries"]), "notes": notes}
    return out


def chosen_settings(rows: list[dict]) -> dict:
    """{index: (build params, search params, tune recall at that setting)} from the last choose_setting row of each index."""
    out = {}
    for r in rows:
        if r["part"] == "part4" and r["experiment"] == "choose_setting":
            out[r["index"]] = (json.loads(r["build_params"]), json.loads(r["search_params"]), float(r["recall_at_10"]))
    return out


def plot(data: dict, shifts: list[float], counts: str, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    (a, b), (c, d) = axes

    def line(ax, index, arm, values, color=None, **kw):
        v = ARMS[arm]
        ax.plot(shifts, values, color=color or STYLE[index]["color"], ls=v["ls"], marker=v["marker"], lw=2, ms=6,
                mec=SURFACE, mew=1, zorder=3, **kw)

    for index in INDEXES:
        control_qps = data[(index, 0.0, "stale")]["qps"]
        for arm in ARMS:
            line(a, index, arm, [data[(index, s, arm)]["recall"] for s in shifts])
            line(b, index, arm, [data[(index, s, arm)]["qps"] / control_qps for s in shifts])
    scanned = {(i, arm): [data[(i, s, arm)]["notes"]["vectors_scanned_per_query"] for s in shifts]
               for i in ("IVF-Flat", "IVF-PQ") for arm in ARMS}
    # IVF-Flat and IVF-PQ train the same k-means lists, so their curves lie on top of each other: draw one then
    same = all(scanned[("IVF-Flat", arm)] == scanned[("IVF-PQ", arm)] for arm in ARMS)
    for arm in ARMS:
        for index in (("IVF-Flat",) if same else ("IVF-Flat", "IVF-PQ")):
            line(c, index, arm, scanned[(index, arm)], color=INK if same else None)
    for arm in ARMS:
        line(d, "IVF-PQ", arm, [data[("IVF-PQ", s, arm)]["notes"]["pq_squared_error_new"] for s in shifts])
    old_error = data[("IVF-PQ", 0.0, "stale")]["notes"]["pq_squared_error_old"]
    d.axhline(old_error, color=INK_MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    d.text(shifts[-1], old_error * 1.25, "error on the old vectors", ha="right", fontsize=8, color=INK_MUTED)

    b.axhline(1.0, color=INK_MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    d.set_yscale("log")
    a.set_ylabel("recall@10 at the fixed setting")
    b.set_ylabel("speed relative to the same index at shift 0")
    c.set_ylabel("vectors scanned per query")
    d.set_ylabel("squared PQ error, new vectors")
    a.set_title("(a) Recall", loc="left", fontsize=10)
    b.set_title("(b) Speed (queries per second)", loc="left", fontsize=10)
    c.set_title("(c) IVF: vectors scanned per query", loc="left", fontsize=10)
    if same:
        c.text(0.98, 0.04, "identical for IVF-Flat and IVF-PQ", transform=c.transAxes, ha="right", fontsize=8,
               color=INK_MUTED)
    d.set_title("(d) IVF-PQ: how badly the codes fit the new vectors", loc="left", fontsize=10)
    for ax in axes.flat:
        ax.set_xticks(shifts)
        ax.set_xlabel("shift (units of a vector's distance to its cluster centre)")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=STYLE[i]["color"], lw=3, label=i) for i in INDEXES]
    handles += [Line2D([], [], color=INK_MUTED, ls=v["ls"], marker=v["marker"], ms=5, lw=1.5, label=v["label"])
                for v in ARMS.values()]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Drifting data: {counts}", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"part4_drift.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="read results_quick/ instead of results/")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or ROOT / ("results_quick" if args.quick else "results")
    rows = results.read_rows(results_dir / "results.csv")
    data = load(rows)
    if not data:
        raise SystemExit("no part4 rows in results.csv - run part4_benchmark.py first")
    shifts = sorted({s for _, s, _ in data})
    settings = chosen_settings(rows)

    setting_body = []
    for index in INDEXES:
        build, search, tune_recall = settings[index]
        build = ", ".join(f"{k}={v}" for k, v in build.items())
        setting_body.append([index, build, ", ".join(f"{k}={v}" for k, v in search.items()), f"{tune_recall:.3f}"])

    summary = []
    for index in INDEXES:
        control = data[(index, 0.0, "stale")]["qps"]
        for shift in shifts:
            for arm in ARMS:
                m = data[(index, shift, arm)]
                n = m["notes"]
                spread = 100 * (max(m["repeats"]) - min(m["repeats"])) / statistics.median(m["repeats"])
                summary.append([index, f"{shift:g}", arm, f"{m['recall']:.3f}", f"{m['qps']:.0f}", f"{spread:.0f}%",
                                f"{m['qps'] / control:.2f}",
                                f"{m['qps'] / data[(index, shift, 'retrained')]['qps']:.2f}",
                                f"{n['vectors_scanned_per_query']:.0f}" if "vectors_scanned_per_query" in n else "-",
                                str(n["max_list_size"]) if "max_list_size" in n else "-",
                                f"{n['pq_squared_error_new']:.3g}" if "pq_squared_error_new" in n else "-"])

    cost = []
    for index in INDEXES:
        stale = [data[(index, s, "stale")] for s in shifts]
        retrained = [data[(index, s, "retrained")] for s in shifts]
        old_build = stale[0]["notes"]["old_build_seconds"]
        add = statistics.median(m["seconds"] for m in stale)
        rebuild = statistics.median(m["seconds"] for m in retrained)
        cost.append([index, f"{old_build:.1f}", f"{add:.1f}", f"{rebuild:.1f}", f"{rebuild / add:.0f}x"])

    tables = results_dir / "tables"
    write_table(tables, "part4_settings", ["index", "build setting", "search setting (chosen at shift 0)", "tune recall"],
                setting_body, "The search setting is the smallest that reaches recall 0.95 on the tune queries of the control, "
                "measured with the stale index at shift 0; it is then kept fixed.")
    write_table(tables, "part4_summary",
                ["index", "shift", "version", "recall@10", "QPS", "spread of the repeats", "QPS / QPS of the stale index at shift 0",
                 "QPS / QPS of the retrained index at the same shift", "vectors scanned per query", "largest list", "PQ squared error on new vectors"], summary,
                f"Report queries, one at a time on one thread (median of {len(next(iter(data.values()))['repeats'])} repeats; 'spread' is "
                "(slowest - fastest) / median of those repeats). Shift 0 is the control. A shift is measured in "
                "units of a typical vector's distance to its own cluster centre. Part of a speed change from shift 0 comes from the data "
                "itself (the same change shows in the retrained version); the column 'QPS / QPS of the retrained index at the same shift' isolates the effect of staleness.")
    write_table(tables, "part4_cost",
                ["index", "build on the old vectors (s)", "adding the new vectors (s, median over shifts)",
                 "retrain / rebuild on all vectors (s, median over shifts)", "rebuild / add"], cost,
                "Builds use all physical cores. HNSW builds are multi-threaded, so their times (and graphs) vary a little between runs.")
    notes = data[("HNSW", 0.0, "stale")]["notes"]
    counts = (f"{notes['n_old']:,} old + {notes['n_new']:,} new vectors, 100 clusters in 64 dimensions, "
              f"{data[('HNSW', 0.0, 'stale')]['n_queries']} report queries")
    plot(data, shifts, counts, results_dir / "figures")
    for name in ("part4_settings", "part4_summary", "part4_cost"):
        print((tables / f"{name}.md").read_text())
    print(f"figure in {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
