#!/usr/bin/env python3
"""CSV logging: one file per run, plus an appended summary across runs."""

import csv
from pathlib import Path

DEFAULT_RUN_DIR = Path("runs")

TICK_COLUMNS = [
    "tick", "hero_row", "hero_col", "action", "enemies_alive", "wrecks",
    "dist_to_goal", "dist_nearest_enemy", "path_len", "replanned",
    "nodes_expanded", "plan_ms", "teleports_left", "event",
]

SUMMARY_COLUMNS = [
    "seed", "outcome", "cause", "ticks", "path_len", "enemies_destroyed",
    "enemies_alive", "teleports_used", "total_plan_ms", "rho_measured",
    "goal_reachable_at_init", "planner", "enemy_spawn_radius", "adjacency_death",
]


def _tick_row(record):
    return {
        "tick": record.tick,
        "hero_row": record.hero[0],
        "hero_col": record.hero[1],
        "action": record.action,
        "enemies_alive": record.enemies_alive,
        "wrecks": record.wrecks,
        "dist_to_goal": record.dist_to_goal,
        "dist_nearest_enemy": record.dist_nearest_enemy,
        "path_len": record.path_len,
        "replanned": int(record.replanned),
        "nodes_expanded": record.nodes_expanded,
        "plan_ms": record.plan_ms,
        "teleports_left": record.teleports_left,
        "event": record.event,
    }


def write_tick_log(result, run_dir=DEFAULT_RUN_DIR):
    """Write the per-tick log for one run. Returns the path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"run-{result.seed}.csv"

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TICK_COLUMNS)
        writer.writeheader()
        for record in result.records:
            writer.writerow(_tick_row(record))
    return path


def append_summary(result, planner_name, config, run_dir=DEFAULT_RUN_DIR):
    """Append one row to summary.csv, writing the header if new."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.csv"
    is_new = not path.exists()

    row = {
        "seed": result.seed,
        "outcome": result.outcome.value,
        "cause": result.cause,
        "ticks": result.ticks,
        "path_len": result.path_len,
        "enemies_destroyed": result.enemies_destroyed,
        "enemies_alive": result.enemies_alive,
        "teleports_used": result.teleports_used,
        "total_plan_ms": result.total_plan_ms,
        "rho_measured": result.rho_measured,
        "goal_reachable_at_init": int(result.goal_reachable_at_init),
        "planner": planner_name,
        "enemy_spawn_radius": config.get("enemy_spawn_radius"),
        "adjacency_death": int(bool(config.get("adjacency_death"))),
    }

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return path
