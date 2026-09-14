"""Game loop, outcomes and termination."""

import random

import numpy as np
import pytest

from flatland.entities import Action
from flatland.game import Game, Outcome
from flatland.grid import OBSTACLE, Grid
from flatland.planners.base import InvalidPlannerDebug, Planner, PlannerDebug
from flatland.planners.baseline import BaselinePlanner
from flatland.worldgen import Scenario, build_scenario


def make_scenario(rows, cols, hero, goal, obstacles=(), enemies=()):
    static = np.zeros((rows, cols), dtype=np.uint8)
    for cell in obstacles:
        static[cell[0], cell[1]] = OBSTACLE
    grid = Grid(static, tuple(goal))
    return Scenario(
        grid=grid,
        hero=tuple(hero),
        enemies=[tuple(c) for c in enemies],
        rho_measured=0.0,
        goal_reachable_at_init=grid.bfs(tuple(hero), tuple(goal)) is not None,
    )


class ScriptedPlanner(Planner):
    """Emits a fixed list of actions, then waits forever."""

    name = "scripted"

    def __init__(self, actions):
        self.actions = list(actions)
        self.seen = 0

    def act(self, obs):
        self.seen += 1
        if self.actions:
            return self.actions.pop(0)
        return Action.wait()

    def debug(self):
        return PlannerDebug(label="scripted", nodes_expanded=self.seen)


class ConcedingPlanner(Planner):
    name = "conceder"

    def act(self, obs):
        return None


def test_reaching_the_goal_wins():
    scenario = make_scenario(5, 5, hero=(0, 0), goal=(0, 2))
    game = Game(scenario, ScriptedPlanner([
        Action.move_to((0, 1)), Action.move_to((0, 2)),
    ]), random.Random(0))

    result = game.run()

    assert result.outcome is Outcome.WIN
    assert result.ticks == 2
    assert result.path_len == 2
    assert result.cause == "hero reached the goal"


def test_reaching_the_goal_beats_an_adjacent_enemy_on_the_same_tick():
    """The hero got there first, so the enemy phase must not run."""
    scenario = make_scenario(5, 5, hero=(0, 0), goal=(0, 1), enemies=[(2, 1)])
    game = Game(scenario, ScriptedPlanner([Action.move_to((0, 1))]), random.Random(0))

    result = game.run()

    assert result.outcome is Outcome.WIN
    assert result.enemies_alive == 1
    # The enemy never moved from its spawn.
    assert game.enemies.members[0].cell == (2, 1)


def test_being_contacted_loses():
    scenario = make_scenario(5, 5, hero=(0, 0), goal=(4, 4), enemies=[(0, 2)])
    game = Game(scenario, ScriptedPlanner([Action.wait(), Action.wait()]),
                random.Random(0))

    result = game.run()

    assert result.outcome is Outcome.LOSS
    assert "contacted hero" in result.cause
    assert result.ticks == 2


def test_conceding_planner_yields_no_path_without_consuming_a_tick():
    scenario = make_scenario(5, 5, hero=(0, 0), goal=(4, 4))
    game = Game(scenario, ConcedingPlanner(), random.Random(0))

    result = game.run()

    assert result.outcome is Outcome.NO_PATH
    assert result.ticks == 0
    assert result.cause == "planner reported no path"
    # The concession is still logged, so the CSV shows why the run ended.
    assert result.records[-1].action == "CONCEDE"


def test_tick_cap_produces_timeout():
    scenario = make_scenario(5, 5, hero=(0, 0), goal=(4, 4))
    game = Game(scenario, ScriptedPlanner([]), random.Random(0), max_ticks=7)

    result = game.run()

    assert result.outcome is Outcome.TIMEOUT
    assert result.ticks == 7
    assert "tick cap 7" in result.cause


def test_stepping_after_termination_returns_none():
    scenario = make_scenario(3, 3, hero=(0, 0), goal=(0, 1))
    game = Game(scenario, ScriptedPlanner([Action.move_to((0, 1))]), random.Random(0))
    game.run()

    assert game.step() is None


def test_interrupt_marks_the_run_and_still_returns_a_result():
    scenario = make_scenario(5, 5, hero=(0, 0), goal=(4, 4))
    game = Game(scenario, ScriptedPlanner([]), random.Random(0), max_ticks=50)
    game.step()
    game.interrupt()

    result = game.result()
    assert result.outcome is Outcome.INTERRUPTED
    assert result.ticks == 1


def test_one_record_per_tick_with_populated_fields():
    scenario = make_scenario(6, 6, hero=(0, 0), goal=(0, 3), enemies=[(5, 5)])
    game = Game(scenario, ScriptedPlanner([
        Action.move_to((0, 1)), Action.move_to((0, 2)), Action.move_to((0, 3)),
    ]), random.Random(0))

    result = game.run()

    assert len(result.records) == result.ticks == 3
    first = result.records[0]
    assert first.tick == 1
    assert first.hero == (0, 1)
    assert first.dist_to_goal == 2
    assert first.dist_nearest_enemy is not None
    assert first.teleports_left == 5
    assert first.nodes_expanded == 1  # ScriptedPlanner reports its call count
    assert result.total_plan_ms >= 0.0


def test_teleport_is_reflected_in_the_result():
    scenario = make_scenario(6, 6, hero=(0, 0), goal=(5, 5))
    game = Game(scenario, ScriptedPlanner([Action.teleport()]), random.Random(0),
                max_ticks=1)

    result = game.run()

    assert result.teleports_used == 1
    assert any("teleport" in record.event for record in result.records)


def test_observation_grid_is_read_only():
    scenario = make_scenario(5, 5, hero=(0, 0), goal=(4, 4))
    game = Game(scenario, ScriptedPlanner([]), random.Random(0))
    obs = game.observe()

    with pytest.raises(TypeError, match="read-only"):
        obs.grid.add_obstacle((1, 1))
    with pytest.raises(ValueError):
        obs.static[0, 0] = 1


def test_observation_reports_live_state():
    scenario = make_scenario(6, 6, hero=(1, 1), goal=(5, 5), enemies=[(3, 3), (4, 4)])
    game = Game(scenario, ScriptedPlanner([]), random.Random(0))
    obs = game.observe()

    assert obs.hero == (1, 1)
    assert obs.goal == (5, 5)
    assert set(obs.enemies) == {(3, 3), (4, 4)}
    assert obs.teleports_remaining == 5
    assert (obs.rows, obs.cols) == (6, 6)
    assert obs.grid.bfs(obs.hero, obs.goal) is not None


def test_wrecks_permanently_block_the_hero():
    """Two enemies collide, and the resulting obstacle changes the graph."""
    scenario = make_scenario(8, 8, hero=(0, 4), goal=(7, 7),
                             enemies=[(2, 4), (1, 3)])
    game = Game(scenario, ScriptedPlanner([Action.wait()]), random.Random(0),
                max_ticks=1)
    game.run()

    assert game.enemies.destroyed_count == 2
    assert game.grid.wreck_mask.sum() >= 1
    wrecked = list(zip(*np.nonzero(game.grid.wreck_mask)))
    assert all(game.grid.is_obstacle(cell) for cell in wrecked)


# --- integration --------------------------------------------------------------


def test_baseline_reaches_a_terminal_outcome_on_every_seed():
    for seed in range(15):
        rng = random.Random(seed)
        game = Game(build_scenario(rng), BaselinePlanner(), rng, seed=seed)
        result = game.run()

        assert result.outcome.is_terminal
        assert result.outcome is not Outcome.RUNNING
        assert len(result.records) >= 1


def test_seed_5_is_a_genuinely_unsolvable_map():
    """A real unsolvable scenario, not a synthetic fixture.

    The hero spawns in a two-cell pocket, so the goal is unreachable by walking
    and the teleporter is the only way out. Useful as a demo case.
    """
    scenario = build_scenario(random.Random(5))

    assert scenario.goal_reachable_at_init is False
    assert len(scenario.grid.flood_reachable(scenario.hero)) == 2

    rng = random.Random(5)
    result = Game(build_scenario(rng), BaselinePlanner(), rng, seed=5).run()
    assert result.outcome is Outcome.NO_PATH


def test_runs_are_reproducible_for_a_given_seed():
    def play(seed):
        rng = random.Random(seed)
        return Game(build_scenario(rng), BaselinePlanner(), rng, seed=seed).run()

    first, again = play(1), play(1)

    assert first.outcome == again.outcome
    assert first.ticks == again.ticks
    assert [r.hero for r in first.records] == [r.hero for r in again.records]


class PlaceholderDebugPlanner(BaselinePlanner):
    """Returns the README skeleton's placeholders literally. `...` is Ellipsis."""

    def debug(self):
        return PlannerDebug(path=..., label=..., visited=..., frontier=...,
                            nodes_expanded=..., predicted_enemies=...,
                            cost_field=...)


def test_undrawable_debug_fields_are_dropped_not_raised():
    """The debug channel is instrumentation; a mistake there must not end a run."""
    with pytest.warns(InvalidPlannerDebug, match="Ellipsis"):
        debug = PlannerDebug(path=..., label=...)

    assert debug.path is None
    assert debug.label is None
    assert debug.predicted_enemies == {}


def test_a_run_survives_a_planner_that_emits_placeholder_debug():
    rng = random.Random(1)
    game = Game(build_scenario(rng), PlaceholderDebugPlanner(), rng, seed=1)

    with pytest.warns(InvalidPlannerDebug):
        result = game.run()

    assert result.outcome is Outcome.WIN
    assert all(record.path_len is None for record in result.records)


def test_valid_debug_fields_survive_untouched():
    field = np.zeros((4, 4))
    debug = PlannerDebug(path=[(0, 0), (0, 1)], visited={(1, 1)}, frontier=[(2, 2)],
                         label="a*", nodes_expanded=7, cost_field=field,
                         predicted_enemies={1: {(3, 3)}})

    assert debug.path == [(0, 0), (0, 1)]
    assert debug.visited == {(1, 1)} and debug.frontier == [(2, 2)]
    assert debug.label == "a*" and debug.nodes_expanded == 7
    assert debug.cost_field is field
    assert debug.predicted_enemies == {1: {(3, 3)}}
