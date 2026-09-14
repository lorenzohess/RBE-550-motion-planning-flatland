"""The parameter sweep: axis subsampling, cell enumeration, and one run's row.

The end-to-end verification is ``python -m flatland.sweep --smoke``. These cover
the pure logic that decides *what* gets run, which is where a silent mistake
would poison a whole sweep without anything looking wrong.
"""

from dataclasses import fields

import pytest

from flatland.config import PLANNER_IDS, PlannerConfig, WorldConfig
from flatland.sweep import (
    AXES,
    TIMING_COLUMNS,
    WORLD_FIELDS,
    columns,
    plan,
    run_cell,
    subsample,
    tasks,
)


# --- subsampling -------------------------------------------------------------


def test_subsample_keeps_both_endpoints_and_the_default():
    values = (1, 2, 3, 4, 5, 6, 7)
    picked = subsample(values, 3, default=4)
    assert picked[0] == 1 and picked[-1] == 7
    assert 4 in picked


def test_subsample_fills_to_the_requested_count_when_the_default_is_an_endpoint():
    # The required set collapses to two, so the fill has to make up the third.
    picked = subsample((0, 1, 2, 3, 4, 5), 3, default=5)
    assert len(picked) == 3
    assert picked[0] == 0 and picked[-1] == 5


def test_subsample_returns_everything_when_asked_for_more_points_than_exist():
    assert subsample((1, 2, 3), 5, default=2) == [1, 2, 3]
    assert subsample((1, 2, 3), None, default=2) == [1, 2, 3]


def test_subsample_stays_ordered_and_unique():
    for axis in AXES:
        for count in (3, 5, None):
            source = WorldConfig() if axis.name in WORLD_FIELDS else PlannerConfig()
            picked = subsample(axis.values, count, getattr(source, axis.name))
            assert len(picked) == len(set(picked))
            assert picked == [v for v in axis.values if v in picked]


def test_subsample_rejects_a_default_outside_the_axis():
    with pytest.raises(ValueError, match="not one of"):
        subsample((1, 2, 3), 3, default=9)


@pytest.mark.parametrize(
    "axis_name, expected",
    [
        # experiments.md 3.1 and 3.2. If this fails, either the doc or the axis
        # list moved and the other has to follow.
        ("size", [32, 64, 128]),
        ("enemy_spawn_radius", [8, 20, None]),
        ("teleport_budget", [0, 2, 5]),
        ("enemy_count", [5, 10, 20]),
        ("rho", [0.1, 0.2, 0.3]),
        ("enemy_radius", [4, 8, 16]),
        ("enemy_weight", [1.0, 10.0, 100.0]),
        ("teleport_threshold", [1, 2, 4]),
    ],
)
def test_coarse_values_match_the_documented_table(axis_name, expected):
    axis = next(a for a in AXES if a.name == axis_name)
    source = WorldConfig() if axis_name in WORLD_FIELDS else PlannerConfig()
    assert subsample(axis.values, 3, getattr(source, axis_name)) == expected


# --- cell enumeration --------------------------------------------------------


def test_coarse_plan_matches_the_documented_cell_counts():
    counts = {}
    for cell in plan("coarse"):
        counts[cell.planner.planner] = counts.get(cell.planner.planner, 0) + 1
    assert counts == {
        "baseline": 11,
        "dijk": 11,
        "dijk-teleport": 13,
        "dijk-cost": 15,
    }


def test_every_cell_varies_exactly_one_axis_from_the_default():
    for cell in plan("medium"):
        differences = [
            field.name
            for config, default in (
                (cell.planner, PlannerConfig(planner=cell.planner.planner)),
                (cell.world, WorldConfig()),
            )
            for field in fields(config)
            if getattr(config, field.name) != getattr(default, field.name)
        ]
        if cell.axis == "default":
            assert differences == []
        else:
            assert differences == [cell.axis]


def test_planner_knobs_are_not_swept_for_planners_that_ignore_them():
    swept = {cell.axis for cell in plan("fine", planners=("baseline",))}
    assert not swept & {"enemy_radius", "enemy_weight", "teleport_threshold"}


def test_restricting_to_one_axis_gives_the_whole_curve_including_the_default():
    cells = plan("fine", axis="enemy_weight", planners=("dijk-cost",))
    axis = next(a for a in AXES if a.name == "enemy_weight")
    assert [cell.planner.enemy_weight for cell in cells] == list(axis.values)
    assert {cell.axis for cell in cells} == {"enemy_weight"}


def test_every_cell_sees_the_same_seed_set():
    cells = plan("coarse", planners=("dijk",))
    by_cell = {}
    for planner, world, _axis, seed in tasks(cells, seeds=5):
        by_cell.setdefault((planner, world), []).append(seed)
    assert len(by_cell) == len(cells)
    assert all(seeds == [0, 1, 2, 3, 4] for seeds in by_cell.values())


# --- one run's row -----------------------------------------------------------


def test_columns_cover_every_config_field_so_a_new_axis_needs_no_writer_change():
    header = columns()
    for config in (PlannerConfig, WorldConfig):
        assert {field.name for field in fields(config)} <= set(header)
    assert len(header) == len(set(header))


def test_run_cell_returns_exactly_the_declared_columns():
    row = run_cell((PlannerConfig(planner="dijk"), WorldConfig(size=32), "default", 0))
    assert set(row) == set(columns())
    assert row["outcome"] in {"win", "loss", "timeout", "no_path"}
    assert row["planner_name"] == "dijk"
    assert row["max_ticks"] == 320  # 10 x size, not the 600 default


def test_run_cell_reports_a_failed_run_instead_of_raising():
    # A 32x32 board at rho=0.2 has ~820 free cells, so placement gives up.
    row = run_cell((PlannerConfig(), WorldConfig(size=32, enemy_count=2000), "x", 0))
    assert row["outcome"] == "error"
    assert "RuntimeError" in row["cause"]
    assert set(row) == set(columns())


def test_a_conceded_run_still_reports_the_cost_of_its_one_decision():
    # Seed 140's goal is unreachable, so with no teleports it concedes on tick 0.
    row = run_cell((PlannerConfig(), WorldConfig(teleport_budget=0), "x", 140))
    assert row["outcome"] == "no_path"
    assert row["ticks"] == 0
    assert row["mean_plan_ms"] > 0  # would be a division by zero if taken per tick


def test_reruns_agree_on_everything_except_wall_clock():
    task = (PlannerConfig(planner="dijk-teleport"), WorldConfig(size=32), "default", 3)
    first, second = run_cell(task), run_cell(task)
    for key in columns():
        if key not in TIMING_COLUMNS:
            assert first[key] == second[key], key
