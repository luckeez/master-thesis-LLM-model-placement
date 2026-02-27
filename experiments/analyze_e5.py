"""
Analyze and plot E5 experiment results.

Usage:
    python experiments/analyze_e5.py
    python experiments/analyze_e5.py --input experiments/results/summary-E5.json --outdir experiments/results/e5_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and plot E5 results")
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/results/summary-E5.json",
        help="Path to E5 summary JSON",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/results/e5_analysis",
        help="Output directory for plots and derived metrics",
    )
    return parser.parse_args()


def classify_family(test_name: str) -> str:
    if "_cheap_" in test_name:
        return "cheap"
    if "_hybrid_" in test_name:
        return "hybrid"
    if "_exp_" in test_name:
        return "expensive"
    return "other"


def classify_mode(test_name: str) -> str:
    if "perf++" in test_name:
        return "perf++"
    if "_perf-" in test_name:
        return "perf"
    if "_min-" in test_name or "_min_exp-" in test_name:
        return "min"
    return "other"


def short_label(test_name: str) -> str:
    label = test_name
    if label.startswith("test_E5_"):
        label = label[len("test_E5_") :]
    if "_LLaMa" in label:
        label = label.split("_LLaMa", 1)[0]
    return label


def parse_series(label: str, family: str) -> str:
    if family == "cheap":
        base = label.replace("cheap_min-", "").replace("cheap_perf++-", "").replace("cheap_perf-", "")
        return f"homo-cheap:{base}"
    if family == "expensive":
        base = label.replace("exp_min_exp-", "").replace("exp_perf-", "")
        return f"homo-exp:{base}"
    if family == "hybrid":
        base = label.replace("hybrid_min-", "").replace("hybrid_perf-", "")
        return f"hybrid:{base}"
    return f"other:{label}"


def enrich_rows(rows: List[Dict]) -> List[Dict]:
    enriched: List[Dict] = []
    for row in rows:
        tp = float(row.get("throughput_tokens_s", 0.0))
        cost = float(row.get("rental_cost_usd_hr", 0.0))
        watts = float(row.get("power_watts", 0.0))

        row2 = dict(row)
        row2["family"] = classify_family(row["test_name"])
        row2["mode"] = classify_mode(row["test_name"])
        row2["label"] = short_label(row["test_name"])
        row2["series"] = parse_series(row2["label"], row2["family"])
        row2["tp_per_usd"] = (tp / cost) if cost > 0 else 0.0
        row2["tp_per_watt"] = (tp / watts) if watts > 0 else 0.0
        row2["usd_per_100_tps"] = (100.0 * cost / tp) if tp > 0 else float("inf")
        enriched.append(row2)
    return enriched


def pareto_frontier(rows: List[Dict]) -> List[Dict]:
    candidates = [r for r in rows if r.get("feasible", False)]
    frontier: List[Dict] = []
    for r in candidates:
        dominated = False
        for s in candidates:
            if s is r:
                continue
            better_or_equal_cost = s["rental_cost_usd_hr"] <= r["rental_cost_usd_hr"]
            better_or_equal_tp = s["throughput_tokens_s"] >= r["throughput_tokens_s"]
            strictly_better = (
                s["rental_cost_usd_hr"] < r["rental_cost_usd_hr"]
                or s["throughput_tokens_s"] > r["throughput_tokens_s"]
            )
            if better_or_equal_cost and better_or_equal_tp and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda x: x["rental_cost_usd_hr"])
    return frontier


def plot_cost_vs_tp(rows: List[Dict], frontier: List[Dict], outpath: Path) -> None:
    colors = {"cheap": "tab:blue", "expensive": "tab:red", "hybrid": "tab:green", "other": "tab:gray"}
    markers = {"min": "o", "perf": "s", "perf++": "^", "other": "x"}

    plt.figure(figsize=(11, 7))
    for r in rows:
        if not r.get("feasible", False):
            continue
        plt.scatter(
            r["rental_cost_usd_hr"],
            r["throughput_tokens_s"],
            c=colors.get(r["family"], "tab:gray"),
            marker=markers.get(r["mode"], "x"),
            s=110,
            alpha=0.9,
        )
        plt.annotate(r["label"], (r["rental_cost_usd_hr"], r["throughput_tokens_s"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    if frontier:
        xs = [r["rental_cost_usd_hr"] for r in frontier]
        ys = [r["throughput_tokens_s"] for r in frontier]
        plt.plot(xs, ys, "k--", linewidth=1.5, label="Pareto frontier")

    plt.title("E5: Throughput vs Rental Cost")
    plt.xlabel("Rental cost [USD/hour]")
    plt.ylabel("Throughput [tokens/s]")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_rank_bar(rows: List[Dict], key: str, title: str, ylabel: str, outpath: Path) -> None:
    valid = [r for r in rows if r.get("feasible", False)]
    valid.sort(key=lambda x: x[key], reverse=True)

    labels = [r["label"] for r in valid]
    vals = [r[key] for r in valid]
    colors = ["tab:blue" if r["family"] == "cheap" else "tab:red" if r["family"] == "expensive" else "tab:green" for r in valid]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, vals, color=colors, alpha=0.9)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_line_by_series(rows: List[Dict], key: str, title: str, ylabel: str, outpath: Path) -> None:
    valid = [r for r in rows if r.get("feasible", False)]
    mode_order = ["min", "perf", "perf++"]
    mode_to_x = {m: i for i, m in enumerate(mode_order)}

    series_rows: Dict[str, List[Dict]] = defaultdict(list)
    for r in valid:
        if r["mode"] in mode_to_x:
            series_rows[r["series"]].append(r)

    family_color = {"homo-cheap": "tab:blue", "homo-exp": "tab:red", "hybrid": "tab:green", "other": "tab:gray"}

    plt.figure(figsize=(12, 7))
    for series, srows in sorted(series_rows.items()):
        srows = sorted(srows, key=lambda x: mode_to_x[x["mode"]])
        xs = [mode_to_x[r["mode"]] for r in srows]
        ys = [r[key] for r in srows]
        family_group = series.split(":", 1)[0] if ":" in series else "other"
        plt.plot(xs, ys, marker="o", linewidth=2, color=family_color.get(family_group, "tab:gray"), label=series)

    plt.xticks([0, 1, 2], mode_order)
    plt.title(title)
    plt.xlabel("Topology level")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_line_by_family_avg(rows: List[Dict], key: str, title: str, ylabel: str, outpath: Path) -> None:
    valid = [r for r in rows if r.get("feasible", False)]
    mode_order = ["min", "perf", "perf++"]
    mode_to_x = {m: i for i, m in enumerate(mode_order)}
    family_map = {"cheap": "homo-cheap", "expensive": "homo-exp", "hybrid": "hybrid", "other": "other"}

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in valid:
        mode = r["mode"]
        if mode not in mode_to_x:
            continue
        fam = family_map.get(r["family"], "other")
        grouped[fam][mode].append(r[key])

    colors = {"homo-cheap": "tab:blue", "homo-exp": "tab:red", "hybrid": "tab:green", "other": "tab:gray"}

    plt.figure(figsize=(10, 6))
    for fam in ["homo-cheap", "homo-exp", "hybrid"]:
        if fam not in grouped:
            continue
        xs, ys = [], []
        for m in mode_order:
            vals = grouped[fam].get(m, [])
            if not vals:
                continue
            xs.append(mode_to_x[m])
            ys.append(sum(vals) / len(vals))
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2.5, color=colors[fam], label=f"{fam} (avg)")

    plt.xticks([0, 1, 2], mode_order)
    plt.title(title)
    plt.xlabel("Topology level")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def save_rankings(rows: List[Dict], outpath: Path) -> None:
    valid = [r for r in rows if r.get("feasible", False)]
    payload = {
        "top_throughput": sorted(valid, key=lambda x: x["throughput_tokens_s"], reverse=True)[:5],
        "top_tp_per_usd": sorted(valid, key=lambda x: x["tp_per_usd"], reverse=True)[:5],
        "top_tp_per_watt": sorted(valid, key=lambda x: x["tp_per_watt"], reverse=True)[:5],
    }
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    rows = enrich_rows(rows)
    frontier = pareto_frontier(rows)

    plot_cost_vs_tp(rows, frontier, outdir / "e5_cost_vs_throughput.png")
    plot_rank_bar(rows, "tp_per_usd", "E5: Cost Efficiency (tokens/s per USD/h)", "tokens/s per USD/h", outdir / "e5_tp_per_usd.png")
    plot_rank_bar(rows, "tp_per_watt", "E5: Energy Efficiency (tokens/s per W)", "tokens/s per W", outdir / "e5_tp_per_watt.png")
    plot_line_by_series(
        rows,
        "throughput_tokens_s",
        "E5: Throughput by topology level (each config line)",
        "Throughput [tokens/s]",
        outdir / "e5_line_series_throughput.png",
    )
    plot_line_by_family_avg(
        rows,
        "throughput_tokens_s",
        "E5: Throughput by topology level (family average)",
        "Throughput [tokens/s]",
        outdir / "e5_line_family_throughput.png",
    )
    plot_line_by_series(
        rows,
        "tp_per_usd",
        "E5: Cost efficiency by topology level (each config line)",
        "tokens/s per USD/h",
        outdir / "e5_line_series_tp_per_usd.png",
    )
    plot_line_by_family_avg(
        rows,
        "tp_per_usd",
        "E5: Cost efficiency by topology level (family average)",
        "tokens/s per USD/h",
        outdir / "e5_line_family_tp_per_usd.png",
    )
    save_rankings(rows, outdir / "e5_rankings.json")

    print("Saved:")
    print(f"  - {outdir / 'e5_cost_vs_throughput.png'}")
    print(f"  - {outdir / 'e5_tp_per_usd.png'}")
    print(f"  - {outdir / 'e5_tp_per_watt.png'}")
    print(f"  - {outdir / 'e5_line_series_throughput.png'}")
    print(f"  - {outdir / 'e5_line_family_throughput.png'}")
    print(f"  - {outdir / 'e5_line_series_tp_per_usd.png'}")
    print(f"  - {outdir / 'e5_line_family_tp_per_usd.png'}")
    print(f"  - {outdir / 'e5_rankings.json'}")


if __name__ == "__main__":
    main()
