#!/usr/bin/env python3
"""Report figures from the CSV logs.

    python -m flatland.plots run runs/run-1.csv -o figures/run-1.png
    python -m flatland.plots sweep runs/summary.csv -o figures/sweep.png

Kept out of the simulation loop: matplotlib is only imported when plotting.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _maybe(value, cast=float):
    if value is None or value == "":
        return None
    try:
        return cast(value)
    except ValueError:
        return None


def plot_run(csv_path, out_path):
    """Per-tick planner state for a single run."""
    rows = _read(csv_path)
    if not rows:
        raise SystemExit(f"{csv_path} has no rows")

    ticks = [int(r["tick"]) for r in rows]
    to_goal = [_maybe(r["dist_to_goal"]) for r in rows]
    to_enemy = [_maybe(r["dist_nearest_enemy"]) for r in rows]
    alive = [int(r["enemies_alive"]) for r in rows]
    expanded = [_maybe(r["nodes_expanded"]) for r in rows]
    plan_ms = [_maybe(r["plan_ms"]) for r in rows]

    have_expanded = any(v is not None for v in expanded)
    panels = 3 if have_expanded else 2
    fig, axes = plt.subplots(panels, 1, figsize=(10, 2.6 * panels), sharex=True)

    axes[0].plot(ticks, to_goal, label="distance to goal", color="tab:blue")
    enemy_ticks = [t for t, v in zip(ticks, to_enemy) if v is not None]
    enemy_vals = [v for v in to_enemy if v is not None]
    axes[0].plot(enemy_ticks, enemy_vals, label="distance to nearest enemy",
                 color="tab:red")
    axes[0].axhline(1, color="tab:red", linestyle=":", linewidth=1,
                    label="contact range")
    axes[0].set_ylabel("Manhattan distance")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].step(ticks, alive, where="post", color="tab:purple")
    axes[1].set_ylabel("enemies alive")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.3)

    if have_expanded:
        axes[2].plot(ticks, expanded, color="tab:green", label="nodes expanded")
        axes[2].set_ylabel("nodes expanded")
        axes[2].grid(alpha=0.3)
        twin = axes[2].twinx()
        twin.plot(ticks, plan_ms, color="tab:orange", alpha=0.6, label="plan ms")
        twin.set_ylabel("planning time (ms)")

    # Mark the terminal event.
    final = rows[-1].get("event", "")
    if final:
        axes[0].axvline(ticks[-1], color="black", linestyle="--", linewidth=1)
        # Anchored low, since the legend occupies the top right.
        axes[0].annotate(final, (ticks[-1], axes[0].get_ylim()[0]),
                         xytext=(-4, 14), textcoords="offset points",
                         ha="right", va="bottom", fontsize=8)

    axes[-1].set_xlabel("tick")
    fig.suptitle(f"Planner state — {Path(csv_path).name}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sweep(csv_path, out_path, group_by="enemy_spawn_radius"):
    """Outcome mix across runs, grouped by a parameter column."""
    rows = _read(csv_path)
    if not rows:
        raise SystemExit(f"{csv_path} has no rows")
    if group_by not in rows[0]:
        raise SystemExit(f"column {group_by!r} not in {csv_path}")

    grouped = defaultdict(Counter)
    for row in rows:
        grouped[row[group_by]][row["outcome"]] += 1

    def sort_key(label):
        return (label == "", _maybe(label, float) if label != "" else 0.0)

    labels = sorted(grouped, key=sort_key)
    kinds = ["win", "loss", "timeout", "no_path", "interrupted"]
    colors = {
        "win": "tab:green", "loss": "tab:red", "timeout": "tab:orange",
        "no_path": "tab:gray", "interrupted": "tab:blue",
    }

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 4, 4.5))
    bottoms = [0.0] * len(labels)
    for kind in kinds:
        values = []
        for index, label in enumerate(labels):
            total = sum(grouped[label].values()) or 1
            values.append(100.0 * grouped[label][kind] / total)
        if not any(values):
            continue
        ax.bar(labels, values, bottom=bottoms, label=kind, color=colors[kind])
        for index, value in enumerate(values):
            if value >= 6:
                ax.text(index, bottoms[index] + value / 2, f"{value:.0f}%",
                        ha="center", va="center", fontsize=8, color="white")
            bottoms[index] += value

    counts = [sum(grouped[label].values()) for label in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{label or 'uniform'}\n(n={n})"
                        for label, n in zip(labels, counts)])
    ax.set_ylabel("share of runs (%)")
    ax.set_xlabel(group_by.replace("_", " "))
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Outcome mix")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="flatland.plots")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="per-tick planner state for one run")
    run.add_argument("csv")
    run.add_argument("-o", "--out", default="figures/run.png")

    sweep = sub.add_parser("sweep", help="outcome mix across many runs")
    sweep.add_argument("csv")
    sweep.add_argument("-o", "--out", default="figures/sweep.png")
    sweep.add_argument("--group-by", default="enemy_spawn_radius")

    args = parser.parse_args(argv)
    if args.command == "run":
        print(plot_run(args.csv, args.out))
    else:
        print(plot_sweep(args.csv, args.out, args.group_by))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
