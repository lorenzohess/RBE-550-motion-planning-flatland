#!/usr/bin/env python3
"""Command line entry point."""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

from .game import DEFAULT_MAX_TICKS, Game
from .instrument import append_summary, write_tick_log
from .planners.baseline import BaselinePlanner
from .worldgen import (
    DEFAULT_COLS,
    DEFAULT_ENEMY_COUNT,
    DEFAULT_ENEMY_SPAWN_RADIUS,
    DEFAULT_RHO,
    DEFAULT_ROWS,
    build_scenario,
)

from .config import PLANNER_IDS, PlannerConfig
from .entities import MAX_TELEPORTS

DEFAULT_FPS = 10

# 'hero' predates the variant ids and appears throughout README.md.
PLANNER_ALIASES = {"hero": "dijk-cost"}


def make_planner(config):
    """Build the planner for one run. Accepts a PlannerConfig or a planner id."""
    if isinstance(config, str):
        config = PlannerConfig(planner=PLANNER_ALIASES.get(config, config))
    if config.planner == "baseline":
        return BaselinePlanner()

    try:
        from .planners.hero import HeroPlanner
    except ImportError as exc:
        raise SystemExit(
            "planner 'hero' is not available yet: "
            "create flatland/planners/hero.py defining HeroPlanner "
            f"(subclass flatland.planners.base.Planner).\n  {exc}"
        ) from exc
    return HeroPlanner(config)


def planner_config(args):
    """The PlannerConfig described by the command line."""
    return PlannerConfig(
        planner=PLANNER_ALIASES.get(args.planner, args.planner),
        enemy_radius=args.enemy_radius,
        enemy_weight=args.enemy_weight,
        teleport_threshold=args.teleport_threshold,
    )


def build_game(args, seed):
    rng = random.Random(seed)
    scenario = build_scenario(
        rng,
        rows=args.rows,
        cols=args.cols,
        rho=args.rho,
        enemy_count=args.enemies,
        enemy_spawn_radius=args.enemy_spawn_radius,
    )
    return Game(
        scenario,
        make_planner(planner_config(args)),
        rng,
        max_ticks=args.max_ticks,
        adjacency_death=args.adjacency_death,
        seed=seed,
        teleport_budget=args.teleport_budget,
    )


def play_headless(args, seed):
    return build_game(args, seed).run()


def play_visual(args, seed):
    """Run with rendering. Returns (result, command) where command may be 'restart'."""
    from .render.pygame_view import View
    from .render.recorder import Recorder

    import pygame

    game = build_game(args, seed)
    view = View(
        game.grid.rows,
        game.grid.cols,
        cell_px=args.cell_px,
        title=f"Flatland — seed {seed}",
    )
    recorder = Recorder(args.record, view.size, fps=args.fps) if args.record else None
    clock = pygame.time.Clock()

    paused = args.paused
    command = None
    try:
        while True:
            commands = view.poll()
            if "quit" in commands:
                game.interrupt()
                command = "quit"
                break
            if "restart" in commands:
                command = "restart"
                break
            if "pause" in commands:
                paused = not paused
            single_step = "step" in commands

            view.draw(game, paused=paused)
            if recorder is not None:
                recorder.add(view.surface)

            if game.outcome.is_terminal:
                if not args.live:
                    break
                # Interactive: hold the final frame until the user acts.
                paused = True
                clock.tick(args.fps)
                continue

            if not paused or single_step:
                game.step()

            clock.tick(args.fps)
    finally:
        if recorder is not None:
            try:
                if game.outcome.is_terminal:
                    recorder.hold()
                recorder.close()
            except RuntimeError as exc:
                # The simulation already finished; a failed encode should cost
                # the animation, not the run's results.
                print(f"warning: recording failed: {exc}", file=sys.stderr)
        view.close()

    return game.result(), command


def report(result):
    reachable = "" if result.goal_reachable_at_init else "  (goal unreachable at init)"
    print(
        f"seed {result.seed}: {result.outcome.value.upper()} in {result.ticks} ticks"
        f"{reachable}\n"
        f"  cause             : {result.cause}\n"
        f"  hero moves        : {result.path_len}\n"
        f"  enemies destroyed : {result.enemies_destroyed}/"
        f"{result.enemies_destroyed + result.enemies_alive}\n"
        f"  teleports used    : {result.teleports_used}\n"
        f"  planning time     : {result.total_plan_ms:.1f} ms\n"
        f"  measured density  : {result.rho_measured}"
    )


def run_sweep(args):
    outcomes = Counter()
    config = vars(args)
    planner_name = planner_config(args).name

    for offset in range(args.sweep):
        seed = args.seed + offset
        result = play_headless(args, seed)
        outcomes[result.outcome.value] += 1
        if not args.no_csv:
            append_summary(result, planner_name, config, run_dir=args.run_dir)
        print(
            f"  seed {seed:<5} {result.outcome.value:<9} ticks={result.ticks:<4} "
            f"destroyed={result.enemies_destroyed:<3} "
            f"teleports={result.teleports_used}"
        )

    total = max(args.sweep, 1)
    print(
        f"\n{args.sweep} runs with planner '{planner_name}', "
        f"spawn radius {args.enemy_spawn_radius}:"
    )
    for name in ("win", "loss", "timeout", "no_path", "interrupted"):
        if outcomes[name]:
            print(
                f"  {name:<12} {outcomes[name]:>4}  ({100 * outcomes[name] / total:.0f}%)"
            )
    if not args.no_csv:
        print(f"\nsummary -> {args.run_dir}/summary.csv")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="flatland",
        description="Gridworld Chase: guide an omniscient hero past blind enemies.",
    )

    world = parser.add_argument_group("world")
    world.add_argument(
        "--seed",
        type=int,
        default=0,
        help="master seed; controls terrain, placement and enemy tie-breaks",
    )
    world.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    world.add_argument("--cols", type=int, default=DEFAULT_COLS)
    world.add_argument(
        "--rho",
        type=float,
        default=DEFAULT_RHO,
        help="target obstacle density (default %(default)s)",
    )
    world.add_argument("--enemies", type=int, default=DEFAULT_ENEMY_COUNT)
    world.add_argument(
        "--enemy-spawn-radius",
        type=int,
        default=DEFAULT_ENEMY_SPAWN_RADIUS,
        help="Manhattan radius around the hero for enemy spawns; "
        "use -1 for uniform placement over the whole map (default %(default)s)",
    )

    rules = parser.add_argument_group("rules")
    rules.add_argument(
        "--teleport-budget",
        type=int,
        default=MAX_TELEPORTS,
        help="teleport activations per run (default %(default)s)",
    )
    rules.add_argument(
        "--adjacency-death",
        action="store_true",
        help="an enemy merely adjacent to the hero is fatal (harder variant)",
    )
    rules.add_argument(
        "--max-ticks",
        type=int,
        default=DEFAULT_MAX_TICKS,
        help="tick cap (default %(default)s = 60s at 10 fps)",
    )

    who = parser.add_argument_group("planner")
    who.add_argument(
        "--planner",
        default="baseline",
        choices=(*PLANNER_IDS, "hero"),
        help="'baseline' is enemy-blind BFS; the dijk* variants are "
        "yours ('hero' is an alias for dijk-cost)",
    )
    knobs = PlannerConfig()
    who.add_argument(
        "--enemy-radius",
        type=int,
        default=knobs.enemy_radius,
        help="dijk-cost: cost ramp reaches zero at this distance "
        "(default %(default)s)",
    )
    who.add_argument(
        "--enemy-weight",
        type=float,
        default=knobs.enemy_weight,
        help="dijk-cost: peak enemy-proximity cost (default %(default)s)",
    )
    who.add_argument(
        "--teleport-threshold",
        type=int,
        default=knobs.teleport_threshold,
        help="dijk-teleport: panic when an enemy is this close "
        "(default %(default)s)",
    )

    show = parser.add_argument_group("output")
    show.add_argument(
        "--live", action="store_true", help="open a window and run in real time"
    )
    show.add_argument("--record", metavar="PATH", help="write an mp4 of the run")
    show.add_argument("--fps", type=int, default=DEFAULT_FPS)
    show.add_argument("--cell-px", type=int, default=12)
    show.add_argument(
        "--paused", action="store_true", help="start paused (use with --live)"
    )
    show.add_argument(
        "--sweep",
        type=int,
        metavar="N",
        help="run N consecutive seeds headless and summarize",
    )
    show.add_argument("--run-dir", default="runs")
    show.add_argument("--no-csv", action="store_true", help="skip writing CSV logs")

    args = parser.parse_args(argv)
    if args.enemy_spawn_radius is not None and args.enemy_spawn_radius < 0:
        args.enemy_spawn_radius = None
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.sweep:
        if args.live or args.record:
            raise SystemExit("--sweep cannot be combined with --live or --record")
        return run_sweep(args)

    visual = args.live or args.record
    if args.record and not args.live:
        # Render offscreen so recording works without a display.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    seed = args.seed
    while True:
        if visual:
            result, command = play_visual(args, seed)
        else:
            result, command = play_headless(args, seed), None

        report(result)
        if not args.no_csv:
            log = write_tick_log(result, run_dir=args.run_dir)
            append_summary(
                result,
                planner_config(args).name,
                vars(args),
                run_dir=args.run_dir,
            )
            print(f"  per-tick log      : {log}")
        if args.record and Path(args.record).exists():
            print(f"  animation         : {args.record}")

        if command != "restart":
            break
        print("\nrestarting same seed...\n")

    # A loss is a legitimate game outcome, not a failure of the program.
    return 0


if __name__ == "__main__":
    sys.exit(main())
