"""The rule table from plan.md section 7, one test per row."""

import random

import pytest

from flatland.entities import Action, MAX_TELEPORTS
from flatland.grid import DIRECTIONS, manhattan, step
from flatland.rules import (
    IllegalHeroAction,
    enemy_intent,
    run_tick,
)
from helpers import ScriptedRNG, make_world


# --- enemy collision outcomes ------------------------------------------------


def test_two_enemies_into_one_cell_both_destroyed_one_obstacle():
    # Both enemies are pinned to target (1, 4) by the script.
    grid, hero, enemies = make_world(
        6, 6, hero=(0, 4), enemies=[(2, 4), (1, 3)]
    )
    rng = ScriptedRNG(picks=[(1, 4), (1, 4)])

    events = run_tick(grid, hero, enemies, Action.wait(), rng)

    assert sorted(events.destroyed) == [0, 1]
    assert events.wrecks == [(1, 4)]
    assert grid.is_obstacle((1, 4))
    assert grid.wreck_mask[1, 4]
    assert not events.hero_died
    assert enemies.cells == ()


def test_enemy_into_obstacle_is_destroyed_at_the_target_cell():
    grid, hero, enemies = make_world(
        6, 6, obstacles=[(0, 3)], hero=(0, 0), enemies=[(0, 4)]
    )
    rng = ScriptedRNG()

    events = run_tick(grid, hero, enemies, Action.wait(), rng)

    assert events.destroyed == [0]
    assert events.wrecks == [(0, 3)]
    assert grid.wreck_mask[0, 3]
    assert enemies.cells == ()


def test_enemy_into_a_fresh_wreck_is_destroyed():
    grid, hero, enemies = make_world(6, 6, hero=(0, 0), enemies=[(0, 4)])
    grid.add_obstacle((0, 3), wreck=True)
    rng = ScriptedRNG()

    events = run_tick(grid, hero, enemies, Action.wait(), rng)

    assert events.destroyed == [0]


def test_enemy_into_cell_vacated_this_tick_is_allowed():
    # Hero east of both; each enemy steps east, the trailing one into the cell
    # the leading one just left.
    grid, hero, enemies = make_world(
        6, 6, hero=(0, 5), enemies=[(0, 1), (0, 2)]
    )
    rng = ScriptedRNG()

    events = run_tick(grid, hero, enemies, Action.wait(), rng)

    assert events.destroyed == []
    assert enemies.cells == ((0, 2), (0, 3))


def test_wreck_created_this_tick_does_not_cascade_within_the_tick():
    # Enemies 0 and 1 collide at (1, 4) creating a wreck there; enemy 2 targets
    # that same cell's *neighbour* and must be unaffected this tick.
    grid, hero, enemies = make_world(
        8, 8, hero=(0, 4), enemies=[(2, 4), (1, 3), (3, 5)]
    )
    rng = ScriptedRNG(picks=[(1, 4), (1, 4), (2, 5)])

    events = run_tick(grid, hero, enemies, Action.wait(), rng)

    assert sorted(events.destroyed) == [0, 1]
    assert enemies.cells == ((2, 5),)


def test_enemy_reaching_hero_kills_it():
    grid, hero, enemies = make_world(6, 6, hero=(0, 4), enemies=[(0, 5)])
    rng = ScriptedRNG()

    events = run_tick(grid, hero, enemies, Action.wait(), rng)

    assert events.hero_died
    assert "contacted hero" in events.cause


# --- hero actions ------------------------------------------------------------


def test_hero_moving_onto_an_enemy_dies_and_enemies_do_not_act():
    grid, hero, enemies = make_world(6, 6, hero=(0, 4), enemies=[(0, 5), (3, 3)])
    rng = ScriptedRNG()

    events = run_tick(grid, hero, enemies, Action.move_to((0, 5)), rng)

    assert events.hero_died
    assert events.cause == "hero moved onto enemy"
    # The far enemy never moved, proving the enemy phase was skipped.
    assert enemies.members[1].cell == (3, 3)


def test_hero_wait_advances_the_tick():
    grid, hero, enemies = make_world(6, 6, hero=(0, 0), enemies=[(5, 5)])
    events = run_tick(grid, hero, enemies, Action.wait(), ScriptedRNG())

    assert hero.cell == (0, 0)
    assert not events.hero_died
    assert enemies.members[0].cell != (5, 5)


@pytest.mark.parametrize(
    "target, reason",
    [
        ((-1, 0), "out of bounds"),
        ((2, 2), "not 4-adjacent"),
        ((0, 1), "obstacle"),
    ],
)
def test_illegal_hero_move_warns_and_degrades_to_wait(target, reason):
    grid, hero, enemies = make_world(
        6, 6, obstacles=[(0, 1)], hero=(0, 0), enemies=[(5, 5)]
    )

    with pytest.warns(IllegalHeroAction, match=reason):
        events = run_tick(grid, hero, enemies, Action.move_to(target), ScriptedRNG())

    assert hero.cell == (0, 0)
    assert events.illegal_action
    # The tick still resolved: the enemy moved.
    assert enemies.members[0].cell != (5, 5)


def test_teleport_moves_hero_consumes_budget_and_costs_a_tick():
    grid, hero, enemies = make_world(6, 6, hero=(0, 0), enemies=[(5, 5)])
    rng = random.Random(0)

    events = run_tick(grid, hero, enemies, Action.teleport(), rng)

    assert events.teleported
    assert hero.teleports_used == 1
    assert hero.teleports_remaining == MAX_TELEPORTS - 1
    assert grid.passable(hero.cell)
    assert enemies.members[0].cell != (5, 5)


def test_teleport_never_lands_on_an_enemy_or_obstacle():
    obstacles = [(r, c) for r in range(6) for c in range(6) if (r, c) not in
                 {(0, 0), (0, 1), (5, 5)}]
    grid, hero, enemies = make_world(
        6, 6, obstacles=obstacles, hero=(0, 0), enemies=[(0, 1)]
    )
    rng = random.Random(1)

    run_tick(grid, hero, enemies, Action.teleport(), rng)

    # (0,1) holds an enemy and everything but (5,5) is obstacle, so that's the
    # only legal destination.
    assert hero.cell == (5, 5)


def test_teleport_budget_exhausts_at_five():
    grid, hero, enemies = make_world(8, 8, hero=(0, 0), enemies=[])
    rng = random.Random(0)

    for _ in range(MAX_TELEPORTS):
        events = run_tick(grid, hero, enemies, Action.teleport(), rng)
        assert events.teleported

    assert hero.teleports_remaining == 0
    with pytest.warns(IllegalHeroAction, match="none remaining"):
        events = run_tick(grid, hero, enemies, Action.teleport(), rng)
    assert not events.teleported
    assert events.illegal_action
    assert hero.teleports_used == MAX_TELEPORTS


# --- adjacency death variant -------------------------------------------------


def test_adjacency_is_not_fatal_by_default():
    grid, hero, enemies = make_world(6, 6, hero=(0, 0), enemies=[(0, 2)])
    events = run_tick(grid, hero, enemies, Action.wait(), ScriptedRNG())

    assert enemies.members[0].cell == (0, 1)
    assert not events.hero_died


def test_adjacency_is_fatal_when_enabled():
    grid, hero, enemies = make_world(6, 6, hero=(0, 0), enemies=[(0, 2)])
    events = run_tick(
        grid, hero, enemies, Action.wait(), ScriptedRNG(), adjacency_death=True
    )

    assert events.hero_died
    assert "adjacent" in events.cause


def test_adjacency_death_checked_before_enemies_move():
    # Enemy already adjacent: the hero dies without the enemy phase running.
    grid, hero, enemies = make_world(6, 6, hero=(0, 0), enemies=[(0, 1)])
    events = run_tick(
        grid, hero, enemies, Action.wait(), ScriptedRNG(), adjacency_death=True
    )

    assert events.hero_died
    assert enemies.members[0].cell == (0, 1)


# --- enemy intent properties -------------------------------------------------


def test_enemy_intent_is_always_in_bounds():
    """A step reducing Manhattan distance to an in-bounds hero stays in bounds.

    This is why the boundary-collision branch in resolve_enemies is unreachable.
    """
    rows = cols = 8
    grid, _, _ = make_world(rows, cols)
    rng = random.Random(0)
    for hero_row in range(rows):
        for hero_col in range(cols):
            for enemy_row in range(rows):
                for enemy_col in range(cols):
                    hero_cell = (hero_row, hero_col)
                    enemy_cell = (enemy_row, enemy_col)
                    target = enemy_intent(grid, enemy_cell, hero_cell, rng)
                    assert grid.in_bounds(target), (enemy_cell, hero_cell, target)


def test_enemy_intent_strictly_reduces_distance_or_stands_still():
    grid, _, _ = make_world(8, 8)
    rng = random.Random(0)
    for _ in range(500):
        hero_cell = (rng.randrange(8), rng.randrange(8))
        enemy_cell = (rng.randrange(8), rng.randrange(8))
        target = enemy_intent(grid, enemy_cell, hero_cell, rng)
        before = manhattan(enemy_cell, hero_cell)
        after = manhattan(target, hero_cell)
        if enemy_cell == hero_cell:
            assert target == enemy_cell
        else:
            assert after == before - 1


def test_enemy_intent_ignores_obstacles():
    """Blindness to terrain is the core mechanic, not a bug."""
    grid, _, _ = make_world(6, 6, obstacles=[(0, 3)])
    target = enemy_intent(grid, (0, 4), (0, 0), random.Random(0))
    assert target == (0, 3)
    assert grid.is_obstacle(target)


def test_every_living_enemy_always_has_a_move():
    grid, _, _ = make_world(8, 8)
    for direction in DIRECTIONS:
        cell = step((4, 4), direction)
        assert enemy_intent(grid, cell, (4, 4), random.Random(0)) == (4, 4)
