#!/usr/bin/env python3
"""Report figures from a sweep CSV.

    python -m flatland.figures runs/sweep-medium.csv -o figures

Distinct from :mod:`flatland.plots`, which reads per-run and per-tick logs.
Everything here reads the tidy one-row-per-run format that ``flatland.sweep``
writes, and saves straight to PNG -- no window, so this works over ssh.

Each figure answers one question, and the name says which.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLANNERS = ("baseline", "dijk", "dijk-teleport", "dijk-cost")

PLANNER_COLORS = {
    "baseline": "tab:gray",
    "dijk": "tab:blue",
    "dijk-teleport": "tab:orange",
    "dijk-cost": "tab:green",
}

# Ordered worst-to-best so the stack reads like a thermometer.
OUTCOME_KINDS = ("win", "loss", "timeout", "no_path")
OUTCOME_COLORS = {
    "win": "tab:green",
    "loss": "tab:red",
    "timeout": "tab:orange",
    "no_path": "tab:gray",
}

AXIS_LABELS = {
    "enemy_spawn_radius": "enemy spawn radius",
    "teleport_budget": "teleport budget",
    "size": "grid size",
    "enemy_count": "enemy count",
    "rho": "obstacle density $\\rho$",
    # Spelled "cost-field radius", never "enemy radius": the world already has an
    # enemy_spawn_radius, and the two are trivial to confuse at a glance.
    "enemy_radius": "cost-field radius (dijk-cost)",
    "enemy_weight": "enemy cost weight (dijk-cost)",
    "teleport_threshold": "teleport threshold (dijk-teleport)",
}

# Filenames normally follow the CSV column. These two do not, for the same
# reason: 'sweep-outcome-enemy-radius' next to 'sweep-outcome-enemy-spawn-radius'
# is an invitation to cite the wrong figure in the report.
FILE_SLUGS = {"enemy_radius": "cost-field-radius"}


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _value_key(value):
    """Sort axis values numerically, with uniform (blank) last."""
    return (value == "", float(value) if value else 0.0)


def _label(value):
    if value == "":
        return "uniform"
    number = float(value)
    return str(int(number)) if number == int(number) else str(number)


def groups_along(rows, axis, planner):
    """Rows for one planner along one axis, keyed by that axis's value.

    The default cell supplies the on-default point. It is stored once and shared
    by every axis, which is exactly what makes this one-factor-at-a-time: every
    curve passes through the same anchor.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["planner"] == planner and row["axis"] in (axis, "default"):
            groups[row[axis]].append(row)
    return {key: groups[key] for key in sorted(groups, key=_value_key)}


def win_rate(group):
    return 100.0 * sum(1 for row in group if row["outcome"] == "win") / len(group)


def median_of(group, column, wins_only=False):
    values = [
        float(row[column])
        for row in group
        if row[column] != "" and (not wins_only or row["outcome"] == "win")
    ]
    return statistics.median(values) if values else float("nan")


def swept_planners(rows, axis):
    """Planners that actually vary along this axis, in display order."""
    return [p for p in PLANNERS if len(groups_along(rows, axis, p)) > 1]


def default_cell(rows, planner):
    return [r for r in rows if r["planner"] == planner and r["axis"] == "default"]


def _save(fig, out_path, rect=None):
    """Lay out and write. ``rect`` reserves a band, e.g. for a figure legend.

    The rect has to be applied here rather than by the caller: a later
    tight_layout() would discard it and drop the legend on top of the axis
    labels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- the outcome-mix bar chart -----------------------------------------------


def plot_outcome_mix(rows, axis, out_path):
    """Stacked win/loss/timeout/no_path shares against one parameter.

    One panel per planner. Shares rather than counts so panels are comparable
    even though n is the same everywhere.
    """
    planners = swept_planners(rows, axis)
    if not planners:
        return None

    # Floor the width: a single-planner axis would otherwise give a figure
    # narrower than its own suptitle, which then renders clipped at both ends.
    width = max(7.0, 3.2 * len(planners) + 1.0)
    fig, axes = plt.subplots(1, len(planners), figsize=(width, 4.4), sharey=True)
    axes = axes if len(planners) > 1 else [axes]

    for panel, planner in zip(axes, planners):
        groups = groups_along(rows, axis, planner)
        labels = [_label(value) for value in groups]
        bottoms = [0.0] * len(groups)

        for kind in OUTCOME_KINDS:
            shares = [
                100.0 * sum(1 for r in group if r["outcome"] == kind) / len(group)
                for group in groups.values()
            ]
            if not any(shares):
                continue  # an outcome nobody hit should not claim a legend entry
            panel.bar(labels, shares, bottom=bottoms, color=OUTCOME_COLORS[kind],
                      label=kind, width=0.7)
            for index, share in enumerate(shares):
                if share >= 8:
                    panel.text(index, bottoms[index] + share / 2, f"{share:.0f}",
                               ha="center", va="center", fontsize=7.5, color="white")
                bottoms[index] += share

        panel.set_title(planner, fontsize=10)
        panel.set_xlabel(AXIS_LABELS.get(axis, axis))
        panel.set_ylim(0, 100)
        panel.tick_params(axis="x", labelsize=8)

    axes[0].set_ylabel("share of runs (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
               frameon=False)
    n = len(next(iter(groups_along(rows, axis, planners[0]).values())))
    fig.suptitle(f"Outcome mix vs {AXIS_LABELS.get(axis, axis)}  (n={n} per bar)")
    return _save(fig, out_path, rect=(0, 0.08, 1, 1))


# --- one figure per finding --------------------------------------------------


def plot_axes_overview(rows, out_path):
    """Win rate against every axis: which knobs matter, and which saturate."""
    axes_to_plot = [a for a in AXIS_LABELS if swept_planners(rows, a)]
    columns = 4
    figure_rows = -(-len(axes_to_plot) // columns)
    fig, grid = plt.subplots(
        figure_rows, columns, figsize=(4.0 * columns, 3.1 * figure_rows), sharey=True
    )
    panels = grid.flatten()

    for panel, axis in zip(panels, axes_to_plot):
        for planner in swept_planners(rows, axis):
            groups = groups_along(rows, axis, planner)
            panel.plot(
                [_label(v) for v in groups],
                [win_rate(g) for g in groups.values()],
                marker="o", markersize=4, color=PLANNER_COLORS[planner], label=planner,
            )
        panel.set_title(AXIS_LABELS.get(axis, axis), fontsize=10)
        panel.set_ylim(0, 102)
        panel.grid(alpha=0.3)
        panel.tick_params(labelsize=8)

    for panel in panels[len(axes_to_plot):]:
        panel.axis("off")
    for index in range(0, len(axes_to_plot), columns):
        panels[index].set_ylabel("win rate (%)")

    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle("Win rate along every swept axis, one factor at a time")
    return _save(fig, out_path, rect=(0, 0.06, 1, 1))


def plot_teleport_budget(rows, out_path):
    """The panic button is bought almost entirely with its first activation."""
    fig, panel = plt.subplots(figsize=(7.2, 4.6))
    axis = "teleport_budget"

    for planner in swept_planners(rows, axis):
        groups = groups_along(rows, axis, planner)
        panel.plot(
            [_label(v) for v in groups],
            [win_rate(g) for g in groups.values()],
            marker="o", color=PLANNER_COLORS[planner], label=planner,
        )

    # Indexed by position rather than by literal value, so this survives a
    # change of resolution or of the axis's value list.
    teleport = list(groups_along(rows, axis, "dijk-teleport").values())
    starved, one = win_rate(teleport[0]), win_rate(teleport[1])
    panel.annotate(
        f"one teleport: {starved:.0f}% $\\rightarrow$ {one:.0f}%",
        xy=(1, one), xytext=(1.5, 70), fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
    )
    panel.annotate(
        "with no budget, dijk-teleport is\nbit-identical to dijk on every seed",
        xy=(0, starved), xytext=(0.15, 22), fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
    )

    panel.set_xlabel(AXIS_LABELS[axis])
    panel.set_ylabel("win rate (%)")
    panel.set_ylim(0, 102)
    panel.grid(alpha=0.3)
    panel.legend(loc="lower right", fontsize=9)
    panel.set_title("Teleport budget: the whole advantage is the first activation")
    return _save(fig, out_path)


def plot_safety_vs_win(rows, out_path):
    """Why win rate alone is the wrong headline once it saturates."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    cells = {planner: default_cell(rows, planner) for planner in PLANNERS}
    names = [p for p in PLANNERS if cells[p]]

    bars = left.bar(names, [win_rate(cells[p]) for p in names],
                    color=[PLANNER_COLORS[p] for p in names], width=0.6)
    for bar, planner in zip(bars, names):
        left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                  f"{win_rate(cells[planner]):.0f}%", ha="center", fontsize=9)
    left.set_ylabel("win rate (%)")
    left.set_ylim(0, 110)
    left.tick_params(axis="x", labelsize=9)
    left.set_title("Win rate ranks dijk-teleport first...")
    left.grid(alpha=0.3, axis="y")

    # Share of runs that ever let an enemy within d. Lower is safer, and it keeps
    # discriminating after win rate has pinned at 100%.
    for planner in names:
        distances = [float(r["min_enemy_distance"]) for r in cells[planner]
                     if r["min_enemy_distance"] != ""]
        span = range(0, 9)
        right.plot(
            list(span),
            [100.0 * sum(1 for v in distances if v <= d) / len(distances) for d in span],
            marker="o", markersize=4, color=PLANNER_COLORS[planner], label=planner,
        )
    # d=0 is the "hero moved onto enemy" loss class: treating enemies as
    # obstacles is what drives it to exactly zero.
    baseline_contact = 100.0 * sum(
        1 for r in cells["baseline"]
        if r["min_enemy_distance"] not in ("",) and float(r["min_enemy_distance"]) == 0
    ) / len(cells["baseline"])
    right.annotate(
        f"{baseline_contact:.0f}% of baseline runs end\nby walking onto an enemy;\n"
        "enemies-as-obstacles zeroes it",
        xy=(0.04, baseline_contact), xytext=(1.35, 7), fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
    )

    right.set_xlabel("distance $d$ (Manhattan)")
    right.set_ylabel("runs that let an enemy within $d$ (%)")
    right.set_ylim(0, 102)
    right.grid(alpha=0.3)
    right.legend(loc="lower right", fontsize=9)
    right.set_title("...but dijk-cost keeps enemies furthest away")

    fig.suptitle("Default scenario, n=100 per planner")
    return _save(fig, out_path)


def plot_density_inversion(rows, out_path):
    """Obstacles help the blind planner and hurt the careful one."""
    fig, panel = plt.subplots(figsize=(7.2, 4.6))
    axis = "rho"

    for planner in swept_planners(rows, axis):
        groups = groups_along(rows, axis, planner)
        panel.plot(
            [_label(v) for v in groups],
            [win_rate(g) for g in groups.values()],
            marker="o", color=PLANNER_COLORS[planner], label=planner,
        )

    # Both annotations sit in the empty band between the planner curves, and
    # point outwards, so the two arrows never cross each other.
    blind = [win_rate(g) for g in groups_along(rows, axis, "baseline").values()]
    careful = [win_rate(g) for g in groups_along(rows, axis, "dijk-cost").values()]
    peak = blind.index(max(blind))  # where density stops helping the blind planner
    panel.annotate("...while the careful planner\nruns out of room to detour",
                   xy=(len(careful) - 1, careful[-1]), xytext=(0.15, 62), fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8))
    panel.annotate("blind enemies wreck themselves\non the obstacle field",
                   xy=(peak, blind[peak]), xytext=(0.9, 8), fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8))

    panel.set_xlabel(AXIS_LABELS[axis])
    panel.set_ylabel("win rate (%)")
    panel.set_ylim(0, 102)
    panel.grid(alpha=0.3)
    panel.set_title("Obstacle density cuts both ways")
    # Below the axes rather than inside them: every in-axes corner here either
    # holds a curve or is where an annotation needs to point.
    handles, labels = panel.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    return _save(fig, out_path, rect=(0, 0.07, 1, 1))


def plot_cost_knobs(rows, out_path):
    """The two dijk-cost knobs, and why only one of them is worth tuning."""
    fig, panels = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for panel, axis in zip(panels, ("enemy_weight", "enemy_radius")):
        groups = groups_along(rows, axis, "dijk-cost")
        labels = [_label(value) for value in groups]

        panel.bar(labels, [win_rate(g) for g in groups.values()],
                  color="tab:green", alpha=0.75, width=0.6, label="win rate")
        panel.set_ylabel("win rate (%)", color="tab:green")
        panel.set_ylim(0, 105)
        panel.set_xlabel(AXIS_LABELS[axis])

        twin = panel.twinx()
        twin.plot(labels, [median_of(g, "min_enemy_distance") for g in groups.values()],
                  marker="o", color="tab:purple", label="median min enemy distance")
        twin.set_ylabel("median min enemy distance", color="tab:purple")
        twin.set_ylim(0, 6)

    panels[0].set_title("Weight: buys safety and wins, saturating at 30")
    panels[1].set_title("Radius: safety flattens, wins drift down")
    fig.suptitle("dijk-cost knobs (the other axis held at its default)")
    return _save(fig, out_path)


def plot_detour_vs_win(rows, out_path):
    """What each planner paid in path length for what it got in win rate."""
    fig, panel = plt.subplots(figsize=(7.2, 4.8))

    detours = []
    teleport_point = None
    for planner in PLANNERS:
        cell = default_cell(rows, planner)
        if not cell:
            continue
        detour = median_of(cell, "detour_ratio", wins_only=True)
        detours.append(detour)
        if planner == "dijk-teleport":
            teleport_point = (detour, win_rate(cell))
        panel.scatter(detour, win_rate(cell), s=140, color=PLANNER_COLORS[planner],
                      zorder=3, label=planner)
        # Label away from the x=1 marker, and inward from whichever edge the
        # point sits against, so nothing lands on the axis border.
        right = detour < 1.0
        panel.annotate(
            planner, (detour, win_rate(cell)),
            xytext=(12 if right else -12, -4), textcoords="offset points",
            ha="left" if right else "right", fontsize=9,
        )

    span = max(detours) - min(detours)
    panel.set_xlim(min(detours) - 0.12 * span, max(detours) + 0.12 * span)
    panel.axvline(1.0, color="black", linestyle=":", linewidth=1)
    panel.annotate("shortest path", (1.0, 4), xytext=(4, 0),
                   textcoords="offset points", fontsize=8, rotation=90)
    if teleport_point is not None:
        # Anchored below the point rather than beside it, so the arrow does not
        # run through that point's own label.
        panel.annotate(
            "< 1 means teleports skipped ground,\nnot a shorter-than-shortest path",
            xy=(teleport_point[0], teleport_point[1] - 6),
            xytext=(teleport_point[0] + 0.03, teleport_point[1] - 26), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
        )

    panel.set_xlabel("median detour ratio on wins  (path length / initial BFS distance)")
    panel.set_ylabel("win rate (%)")
    panel.set_ylim(0, 110)
    panel.grid(alpha=0.3)
    panel.set_title("What safety cost in path length")
    return _save(fig, out_path)


FINDINGS = {
    "axes-overview": plot_axes_overview,
    "teleport-budget": plot_teleport_budget,
    "safety-vs-win": plot_safety_vs_win,
    "density-inversion": plot_density_inversion,
    "cost-knobs": plot_cost_knobs,
    "detour-vs-win": plot_detour_vs_win,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="flatland.figures",
        description="Report figures from a flatland.sweep CSV.",
    )
    parser.add_argument("csv", help="sweep CSV, e.g. runs/sweep-medium.csv")
    parser.add_argument("-o", "--out", default="figures", type=Path,
                        help="output directory (default %(default)s)")
    parser.add_argument("--prefix", default="sweep",
                        help="filename prefix (default %(default)s)")
    args = parser.parse_args(argv)

    rows = read(args.csv)
    if not rows:
        raise SystemExit(f"{args.csv} has no rows")

    written = []
    for axis in AXIS_LABELS:
        slug = FILE_SLUGS.get(axis, axis.replace("_", "-"))
        path = plot_outcome_mix(rows, axis, args.out / f"{args.prefix}-outcome-{slug}.png")
        if path is not None:
            written.append(path)

    for name, draw in FINDINGS.items():
        written.append(draw(rows, args.out / f"{args.prefix}-{name}.png"))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
