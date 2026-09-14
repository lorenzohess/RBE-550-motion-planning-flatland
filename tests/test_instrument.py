"""CSV logging and report figures."""

import csv
import random

from flatland.game import Game
from flatland.instrument import SUMMARY_COLUMNS, TICK_COLUMNS, append_summary, write_tick_log
from flatland.planners.baseline import BaselinePlanner
from flatland.plots import plot_run, plot_sweep
from flatland.worldgen import build_scenario


def play(seed):
    rng = random.Random(seed)
    return Game(build_scenario(rng), BaselinePlanner(), rng, seed=seed).run()


def test_tick_log_has_one_row_per_tick_with_the_declared_columns(tmp_path):
    result = play(1)
    path = write_tick_log(result, run_dir=tmp_path)

    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert path.name == "run-1.csv"
    assert len(rows) == len(result.records)
    assert list(rows[0]) == TICK_COLUMNS
    assert [int(r["tick"]) for r in rows] == [rec.tick for rec in result.records]
    assert int(rows[0]["hero_row"]) == result.records[0].hero[0]


def test_summary_writes_header_once_and_appends(tmp_path):
    config = {"enemy_spawn_radius": 20, "adjacency_death": False}
    for seed in (1, 2, 3):
        path = append_summary(play(seed), "baseline-bfs", config, run_dir=tmp_path)

    with path.open() as handle:
        lines = handle.read().splitlines()
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert path.name == "summary.csv"
    assert len(lines) == 4  # one header plus three runs
    assert list(rows[0]) == SUMMARY_COLUMNS
    assert [int(r["seed"]) for r in rows] == [1, 2, 3]
    assert all(r["planner"] == "baseline-bfs" for r in rows)
    assert all(r["enemy_spawn_radius"] == "20" for r in rows)


def test_summary_records_an_unreachable_goal(tmp_path):
    path = append_summary(play(5), "baseline-bfs", {}, run_dir=tmp_path)
    with path.open() as handle:
        row = next(csv.DictReader(handle))

    assert row["outcome"] == "no_path"
    assert row["goal_reachable_at_init"] == "0"


def test_run_directory_is_created_on_demand(tmp_path):
    target = tmp_path / "nested" / "runs"
    path = write_tick_log(play(1), run_dir=target)
    assert path.exists()


def test_plot_run_writes_a_figure(tmp_path):
    csv_path = write_tick_log(play(1), run_dir=tmp_path)
    out = plot_run(csv_path, tmp_path / "figures" / "run.png")

    assert out.exists() if hasattr(out, "exists") else True
    assert (tmp_path / "figures" / "run.png").stat().st_size > 0


def test_plot_sweep_groups_by_a_parameter(tmp_path):
    for radius in (12, None):
        for seed in (1, 2):
            append_summary(
                play(seed), "baseline-bfs",
                {"enemy_spawn_radius": radius, "adjacency_death": False},
                run_dir=tmp_path,
            )

    plot_sweep(tmp_path / "summary.csv", tmp_path / "sweep.png")
    assert (tmp_path / "sweep.png").stat().st_size > 0
