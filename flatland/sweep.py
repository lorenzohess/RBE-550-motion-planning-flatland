#!/usr/bin/env python3
"""One-factor-at-a-time parameter sweep over the planners.

The design lives in ``experiments.md``. In short: hold every axis at its default,
vary one axis at a time, give every cell the same seed set so configurations are
compared on the same maps, and write one tidy row per run.

    python -m flatland.sweep --smoke
    python -m flatland.sweep --resolution coarse -o runs/sweep-coarse.csv
    python -m flatland.sweep --resolution fine --axis enemy_weight --planner dijk-cost

Run ``--smoke`` before trusting a full sweep. It takes seconds and it is the
regression test for the two bugs the sweep design was written around.
"""

import argparse
import csv
import random
import sys
import time
import warnings
from dataclasses import dataclass, fields, replace
from multiprocessing import Pool
from pathlib import Path

from .cli import make_planner
from .config import PLANNER_IDS, PlannerConfig, WorldConfig
from .game import Game
from .grid import manhattan
from .worldgen import build_scenario

DEFAULT_OUT = Path("runs/sweep.csv")

# Four workers, not eight: nproc reports SMT threads on 4 physical cores and this
# work is CPU-bound. Measured 2.59x at 4 against 2.37x at 8.
DEFAULT_WORKERS = 4

# (points per axis, seeds per cell). Coarse is for *finding* the interesting
# axes; at n=30 the standard error is ~9 points, so never quote a coarse number.
RESOLUTIONS = {
    "coarse": (3, 30),
    "medium": (5, 100),
    "fine": (None, 300),
}

# Rough per-run cost in ms at the default 64x64, spawn radius 20, for --dry-run
# estimates only. Measured, but they move with the scenario, so treat the
# estimate as an order of magnitude.
ESTIMATED_MS = {
    "baseline": 116,
    "dijk": 154,
    "dijk-teleport": 154,
    "dijk-cost": 506,
}


@dataclass(frozen=True)
class Axis:
    """One swept parameter and every value it can take.

    The full ordered list is declared once here; ``--resolution`` subsamples it.
    ``applies_to`` keeps planner knobs off the planners that never read them --
    sweeping ``enemy_weight`` for ``baseline`` would just re-run the same cell.
    """

    name: str
    values: tuple
    applies_to: tuple = PLANNER_IDS


# Swept for all four planners. Note that teleport_budget is included for
# baseline, which never teleports: those cells are redundant by construction and
# act as a cheap control that the axis does nothing to an enemy-blind planner.
SCENARIO_AXES = (
    Axis("size", (32, 40, 48, 64, 80, 96, 128)),
    Axis("enemy_spawn_radius", (8, 10, 12, 16, 20, 25, 30, 40, None)),
    Axis("teleport_budget", (0, 1, 2, 3, 4, 5)),
    Axis("enemy_count", (5, 8, 10, 12, 15, 20)),
    Axis("rho", (0.1, 0.15, 0.2, 0.25, 0.3)),
)

# Swept only for the planner that reads them.
PLANNER_AXES = (
    Axis("enemy_radius", (4, 6, 8, 10, 12, 16), applies_to=("dijk-cost",)),
    Axis("enemy_weight", (1.0, 3.0, 10.0, 30.0, 100.0), applies_to=("dijk-cost",)),
    Axis("teleport_threshold", (1, 2, 3, 4), applies_to=("dijk-teleport",)),
)

AXES = SCENARIO_AXES + PLANNER_AXES
AXIS_NAMES = tuple(axis.name for axis in AXES)

WORLD_FIELDS = frozenset(field.name for field in fields(WorldConfig))

METRIC_COLUMNS = [
    "outcome",
    "cause",
    "ticks",
    "path_len",
    "enemies_destroyed",
    "enemies_alive",
    "teleports_used",
    "total_plan_ms",
    "mean_plan_ms",
    "rho_measured",
    "goal_reachable_at_init",
    "initial_bfs_distance",
    "initial_enemy_distance",
    "min_enemy_distance",
    "detour_ratio",
]

# Wall-clock, so they differ between two runs of the same seed. Everything else
# must not -- see the determinism check in smoke().
TIMING_COLUMNS = ("total_plan_ms", "mean_plan_ms")


def columns():
    """CSV header. Derived from the configs, so adding an axis never edits it."""
    return (
        [field.name for field in fields(PlannerConfig)]
        + ["planner_name"]
        + [field.name for field in fields(WorldConfig)]
        + ["max_ticks", "axis", "seed"]
        + METRIC_COLUMNS
        + ["warnings"]
    )


@dataclass(frozen=True)
class SweepCell:
    """One configuration. Every cell is run against the same seed set."""

    planner: PlannerConfig
    world: WorldConfig
    axis: str


def subsample(values, count, default):
    """``count`` values from ``values``, keeping both endpoints and ``default``.

    The default has to survive subsampling or one-factor-at-a-time has no anchor
    to compare against, and the endpoints have to survive or the axis stops
    answering the question it was added for.
    """
    if default not in values:
        raise ValueError(f"default {default!r} is not one of {values!r}")
    if count is None or count >= len(values):
        return list(values)

    last = len(values) - 1
    chosen = {0, last, values.index(default)}
    # Evenly spaced fill, by integer division rather than rounding so the choice
    # never depends on how .5 is rounded.
    for i in range(count):
        if len(chosen) >= count:
            break
        chosen.add(i * last // max(count - 1, 1))
    # Duplicate candidates collapse into the required set, so top up from the
    # middle outwards if we are still short.
    for index in sorted(range(len(values)), key=lambda i: abs(i - last / 2)):
        if len(chosen) >= count:
            break
        chosen.add(index)
    return [values[index] for index in sorted(chosen)]


def _vary(planner, world, axis, value):
    """The cell that holds everything at its default but ``axis``."""
    if axis.name in WORLD_FIELDS:
        return SweepCell(planner, replace(world, **{axis.name: value}), axis.name)
    return SweepCell(replace(planner, **{axis.name: value}), world, axis.name)


def plan(resolution="medium", axis=None, planners=PLANNER_IDS):
    """Every cell to run.

    Without ``axis``: the default cell plus each off-default value of every
    applicable axis -- roughly 11 cells for baseline and dijk, 13 for
    dijk-teleport, 15 for dijk-cost at coarse resolution.

    With ``axis``: every value of that one axis, the default included, which is
    the complete curve a phase-2 follow-up wants.
    """
    points = RESOLUTIONS[resolution][0]
    cells = []

    for planner_id in planners:
        base_planner = replace(PlannerConfig(), planner=planner_id)
        base_world = WorldConfig()

        if axis is None:
            cells.append(SweepCell(base_planner, base_world, "default"))

        for candidate in AXES:
            if planner_id not in candidate.applies_to:
                continue
            if axis is not None and candidate.name != axis:
                continue

            source = base_world if candidate.name in WORLD_FIELDS else base_planner
            default = getattr(source, candidate.name)
            for value in subsample(candidate.values, points, default):
                if axis is None and value == default:
                    continue  # the default cell already covers this point
                cells.append(_vary(base_planner, base_world, candidate, value))

    return cells


def tasks(cells, seeds):
    """Flatten cells into runs. Every cell sees the same seeds, ``0..seeds-1``."""
    for cell in cells:
        for seed in range(seeds):
            yield (cell.planner, cell.world, cell.axis, seed)


def _play(planner_config, world, seed):
    """Run one game and return the metric half of its row."""
    rng = random.Random(seed)
    scenario = build_scenario(
        rng,
        rows=world.size,
        cols=world.size,
        rho=world.rho,
        enemy_count=world.enemy_count,
        enemy_spawn_radius=world.enemy_spawn_radius,
    )

    # Measure the unobstructed route before the run. Wrecks only ever add
    # obstacles, so this is the shortest the journey is ever going to be, and it
    # is the honest denominator for detour_ratio.
    initial_path = scenario.grid.bfs(scenario.hero, scenario.grid.goal)
    initial_bfs_distance = None if initial_path is None else len(initial_path) - 1
    initial_enemy_distance = min(
        (manhattan(cell, scenario.hero) for cell in scenario.enemies), default=None
    )

    result = Game(
        scenario,
        make_planner(planner_config),
        rng,
        max_ticks=world.max_ticks,
        teleport_budget=world.teleport_budget,
        seed=seed,
    ).run()

    # One record per planner call, including the CONCEDE that ends a NO_PATH run,
    # so this is the cost of one decision even when the run lasted zero ticks.
    decisions = len(result.records) or 1
    distances = [
        record.dist_nearest_enemy
        for record in result.records
        if record.dist_nearest_enemy is not None
    ]
    if initial_enemy_distance is not None:
        distances.append(initial_enemy_distance)  # tick 0 has no record of its own

    return {
        "outcome": result.outcome.value,
        "cause": result.cause,
        "ticks": result.ticks,
        "path_len": result.path_len,
        "enemies_destroyed": result.enemies_destroyed,
        "enemies_alive": result.enemies_alive,
        "teleports_used": result.teleports_used,
        "total_plan_ms": result.total_plan_ms,
        "mean_plan_ms": round(result.total_plan_ms / decisions, 4),
        "rho_measured": result.rho_measured,
        "goal_reachable_at_init": int(result.goal_reachable_at_init),
        "initial_bfs_distance": initial_bfs_distance,
        "initial_enemy_distance": initial_enemy_distance,
        "min_enemy_distance": min(distances, default=None),
        # Falsy also covers a zero-length route, which opposite-quadrant
        # placement rules out, but division by it would be a nasty way to find
        # out otherwise.
        "detour_ratio": (
            round(result.path_len / initial_bfs_distance, 4)
            if initial_bfs_distance
            else None
        ),
    }


def run_cell(task):
    """Run one (planner, world, seed) and return a tidy CSV row.

    Warnings are captured into a column rather than printed: 5,000 runs of
    stderr noise would bury the progress line, and "which runs warned" is more
    useful as data anyway. An exception becomes an ``error`` row rather than
    killing the pool -- the sweep runs unattended, and one bad cell should not
    cost the other 4,999 runs.
    """
    planner_config, world, axis, seed = task
    row = {
        **{f.name: getattr(planner_config, f.name) for f in fields(planner_config)},
        "planner_name": planner_config.name,
        **{f.name: getattr(world, f.name) for f in fields(world)},
        "max_ticks": world.max_ticks,
        "axis": axis,
        "seed": seed,
        # Seeded here so that an error row has the same shape as every other
        # row, rather than relying on the CSV writer to fill the gaps.
        **{key: None for key in METRIC_COLUMNS},
        "warnings": "",
    }

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row.update(_play(planner_config, world, seed))
        row["warnings"] = ";".join(
            dict.fromkeys(entry.category.__name__ for entry in caught)
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate, see the docstring
        row["outcome"] = "error"
        row["cause"] = f"{type(exc).__name__}: {exc}"

    return row


def _format_duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _progress(done, total, started, row):
    elapsed = time.perf_counter() - started
    eta = (total - done) * elapsed / done
    line = (
        f"[{100 * done / total:3.0f}%] {done}/{total}  "
        f"elapsed {_format_duration(elapsed)}  eta {_format_duration(eta)}  "
        f"{row['planner_name']} {row['axis']}={row.get(row['axis'], '')}"
    )
    if sys.stderr.isatty():
        print(f"\r{line:<100}", end="", file=sys.stderr, flush=True)
    else:
        print(line, file=sys.stderr, flush=True)


def run(cells, seeds, out_path, workers=DEFAULT_WORKERS, quiet=False):
    """Run every cell against every seed, streaming rows to ``out_path``."""
    queue = list(tasks(cells, seeds))
    total = len(queue)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tasks cost 100-500 ms each, so per-task IPC is noise and small chunks buy a
    # better-balanced tail across the uneven planners.
    chunksize = max(1, min(8, total // (workers * 16)))
    started = time.perf_counter()
    every = max(1, total // 200)
    outcomes = {}

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns(), restval="")
        writer.writeheader()

        def consume(rows):
            for done, row in enumerate(rows, start=1):
                writer.writerow(row)
                outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
                if done % every == 0 or done == total:
                    handle.flush()
                    if not quiet:
                        _progress(done, total, started, row)

        if workers > 1:
            with Pool(workers) as pool:
                consume(pool.imap_unordered(run_cell, queue, chunksize=chunksize))
        else:
            consume(run_cell(task) for task in queue)

    if not quiet and sys.stderr.isatty():
        print(file=sys.stderr)
    return outcomes, time.perf_counter() - started


# --- verification ------------------------------------------------------------

# Verified unreachable in 0..299 at the default world; see experiments.md 1.1.
UNREACHABLE_SEEDS = (5, 115, 140)


def _check(results, name, ok, detail=""):
    results.append((name, bool(ok), detail))
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}{'  ' + detail if detail else ''}")
    return ok


def smoke():
    """The experiments.md section 6 checks. Seconds, and non-negotiable."""
    print("smoke:")
    results = []

    # 1. Every planner id constructs and completes a run, without asking the
    #    rules for something illegal.
    rows = [
        run_cell((PlannerConfig(planner=name), WorldConfig(), "smoke", seed))
        for name in PLANNER_IDS
        for seed in (0, 1)
    ]
    broken = [r for r in rows if r["outcome"] == "error"]
    _check(
        results,
        "every planner completes a run",
        not broken,
        broken[0]["cause"] if broken else f"{len(rows)} runs",
    )
    illegal = [r for r in rows if "IllegalHeroAction" in (r.get("warnings") or "")]
    _check(
        results,
        "no illegal hero actions",
        not illegal,
        f"{len(illegal)} run(s) warned" if illegal else "",
    )

    # 2. The teleport livelock regression: an unreachable goal with no budget
    #    must concede, not spin until the tick cap.
    starved = [
        run_cell(
            (
                PlannerConfig(planner="dijk-cost"),
                WorldConfig(teleport_budget=0),
                "smoke",
                seed,
            )
        )
        for seed in UNREACHABLE_SEEDS
    ]
    bad = [f"seed {r['seed']}={r['outcome']}" for r in starved if r["outcome"] != "no_path"]
    _check(
        results,
        "teleport_budget=0 concedes instead of livelocking",
        not bad,
        ", ".join(bad) if bad else f"seeds {UNREACHABLE_SEEDS} -> no_path",
    )

    # 3. The planner-name regression: variants must be distinguishable in the CSV.
    names = {r["planner_name"] for r in rows}
    _check(
        results,
        "planner names are distinct",
        len(names) == len(PLANNER_IDS),
        ", ".join(sorted(names)),
    )

    # 4. Every configured axis value builds a scenario. Guards against a future
    #    axis value that makes place_hero_and_goal fail for want of a free cell.
    unbuildable = []
    for axis in AXES:
        for value in axis.values:
            if axis.name not in WORLD_FIELDS:
                continue
            world = replace(WorldConfig(), **{axis.name: value})
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    build_scenario(
                        random.Random(0),
                        rows=world.size,
                        cols=world.size,
                        rho=world.rho,
                        enemy_count=world.enemy_count,
                        enemy_spawn_radius=world.enemy_spawn_radius,
                    )
            except Exception as exc:  # noqa: BLE001 -- reporting, not handling
                unbuildable.append(f"{axis.name}={value}: {exc}")
    _check(
        results,
        "every axis value builds a scenario",
        not unbuildable,
        "; ".join(unbuildable) if unbuildable else "",
    )

    # 5. The new metrics are populated on a run that actually finished.
    won = next((r for r in rows if r["outcome"] == "win"), None)
    missing = [
        key
        for key in ("min_enemy_distance", "mean_plan_ms", "detour_ratio")
        if won is not None and won.get(key) is None
    ]
    _check(
        results,
        "metrics populated on a winning run",
        won is not None and not missing,
        "no winning run in the sample" if won is None else ", ".join(missing),
    )

    # 6. Same seed, same result. Timing is wall-clock, so it is excluded.
    task = (PlannerConfig(), WorldConfig(), "smoke", 0)
    first, second = run_cell(task), run_cell(task)
    drift = [
        key
        for key in first
        if key not in TIMING_COLUMNS and first[key] != second[key]
    ]
    _check(results, "runs are deterministic", not drift, ", ".join(drift))

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


# --- entry point -------------------------------------------------------------


def dry_run(cells, seeds, workers):
    """Print the plan and a cost estimate instead of running it."""
    per_planner = {}
    for cell in cells:
        per_planner.setdefault(cell.planner.planner, []).append(cell)

    print(f"{len(cells)} cells x {seeds} seeds = {len(cells) * seeds} runs")
    estimate = 0.0
    for name, group in per_planner.items():
        runs = len(group) * seeds
        estimate += runs * ESTIMATED_MS[name] / 1000.0
        print(f"  {name:<14} {len(group):>3} cells  {runs:>6} runs")

    speedup = 2.59 if workers >= 4 else workers
    print(
        f"\nrough estimate at the default board: "
        f"{_format_duration(estimate / speedup)} on {workers} workers"
    )
    axes = sorted({cell.axis for cell in cells})
    print(f"axes: {', '.join(axes)}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="flatland.sweep",
        description="One-factor-at-a-time parameter sweep over the planners.",
    )
    parser.add_argument(
        "--resolution",
        choices=tuple(RESOLUTIONS),
        default="medium",
        help="points per axis and seeds per cell (default %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        help="override the seed count for this resolution",
    )
    parser.add_argument(
        "--axis",
        choices=AXIS_NAMES,
        help="sweep one axis over all its values, for a phase-2 follow-up",
    )
    parser.add_argument(
        "--planner",
        action="append",
        choices=PLANNER_IDS,
        help="restrict to one planner; repeatable (default: all four)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="processes; 1 runs serially (default %(default)s)",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT,
        help="CSV to write, overwritten if it exists (default %(default)s)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="run the verification checks and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and a cost estimate, then exit",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        return smoke()

    planners = tuple(dict.fromkeys(args.planner)) if args.planner else PLANNER_IDS
    cells = plan(args.resolution, axis=args.axis, planners=planners)
    seeds = args.seeds if args.seeds is not None else RESOLUTIONS[args.resolution][1]

    if not cells:
        # The usual cause is a planner knob paired with a planner that ignores
        # it, e.g. --axis enemy_weight --planner baseline.
        raise SystemExit(
            f"no cells: axis {args.axis!r} does not apply to {', '.join(planners)}"
        )

    if args.dry_run:
        return dry_run(cells, seeds, args.workers)

    outcomes, elapsed = run(cells, seeds, args.out, args.workers, args.quiet)
    total = sum(outcomes.values())
    print(f"\n{total} runs in {_format_duration(elapsed)} -> {args.out}")
    for name in sorted(outcomes, key=lambda key: -outcomes[key]):
        print(f"  {name:<12} {outcomes[name]:>6}  ({100 * outcomes[name] / total:.1f}%)")
    if outcomes.get("error"):
        print("\nthere were error rows; grep the CSV for outcome=error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
