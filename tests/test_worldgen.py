"""Obstacle generation and initial placement."""

import random

import numpy as np
import pytest

from obstacleworld.obstacleworld.Obstacle import (
    Itet,
    Ltet,
    Stet,
    Ttet,
    normalize,
    rotations,
)
from obstacleworld.obstacleworld.World import TETS

from flatland.grid import manhattan
from flatland.worldgen import (
    DEFAULT_ENEMY_COUNT,
    build_scenario,
    generate_static,
    place_enemies,
)


# --- obstacleworld fixes -----------------------------------------------------


@pytest.mark.parametrize(
    "tet, expected",
    [(Ltet, 4), (Itet, 2), (Stet, 2), (Ttet, 4)],
)
def test_rotation_counts_match_tetromino_symmetry(tet, expected):
    assert len(rotations(tet.cells)) == expected


def test_tets_holds_every_distinct_orientation():
    assert len(TETS) == 12
    shapes = [normalize(t.cells) for t in TETS]
    assert len(set(shapes)) == 12


def test_rotations_preserve_cell_count_and_are_normalized():
    for tet in (Ltet, Itet, Stet, Ttet):
        for shape in rotations(tet.cells):
            assert len(shape) == 4
            assert min(c[0] for c in shape) == 0
            assert min(c[1] for c in shape) == 0


def test_generate_static_shape_and_density():
    static, coverage = generate_static(random.Random(0), 64, 64, 0.2)
    assert static.shape == (64, 64)
    # The generator places whole tetrominoes until coverage is reached, so it
    # lands slightly above the nominal density.
    assert 0.2 <= coverage < 0.21
    assert np.isclose(coverage, np.count_nonzero(static) / static.size)


def test_generation_is_reproducible_and_seed_sensitive():
    first, _ = generate_static(random.Random(7), 64, 64, 0.2)
    again, _ = generate_static(random.Random(7), 64, 64, 0.2)
    other, _ = generate_static(random.Random(8), 64, 64, 0.2)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_generation_does_not_disturb_global_random_state():
    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    generate_static(random.Random(99), 64, 64, 0.2)
    assert random.random() == expected


def test_non_square_dimensions_are_respected():
    static, _ = generate_static(random.Random(0), 32, 48, 0.2)
    assert static.shape == (32, 48)


# --- placement ---------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_scenario_placement_invariants(seed):
    scenario = build_scenario(random.Random(seed))
    grid = scenario.grid

    assert grid.passable(scenario.hero)
    assert grid.passable(grid.goal)
    assert scenario.hero != grid.goal
    assert len(scenario.enemies) == DEFAULT_ENEMY_COUNT
    assert len(set(scenario.enemies)) == DEFAULT_ENEMY_COUNT

    for enemy in scenario.enemies:
        assert grid.passable(enemy)
        assert enemy != grid.goal
        assert manhattan(enemy, scenario.hero) > 1, "enemy adjacent at spawn"
        assert manhattan(enemy, scenario.hero) <= 20, "enemy outside spawn radius"


@pytest.mark.parametrize("seed", range(12))
def test_hero_and_goal_occupy_diagonally_opposite_quadrants(seed):
    scenario = build_scenario(random.Random(seed))
    hero_half = (scenario.hero[0] >= 32, scenario.hero[1] >= 32)
    goal_half = (scenario.grid.goal[0] >= 32, scenario.grid.goal[1] >= 32)

    assert hero_half == (not goal_half[0], not goal_half[1])
    # Opposite quadrants guarantee a non-trivial journey.
    assert manhattan(scenario.hero, scenario.grid.goal) >= 2


def test_scenario_is_reproducible():
    first = build_scenario(random.Random(3))
    again = build_scenario(random.Random(3))

    assert np.array_equal(first.grid.static, again.grid.static)
    assert first.hero == again.hero
    assert first.grid.goal == again.grid.goal
    assert first.enemies == again.enemies


def test_uniform_spawn_places_enemies_far_away():
    """The spec-literal comparison mode."""
    scenario = build_scenario(random.Random(0), enemy_spawn_radius=None)
    distances = [manhattan(e, scenario.hero) for e in scenario.enemies]
    assert all(d > 1 for d in distances)
    assert max(distances) > 20, "uniform spawn should reach beyond the default radius"


def test_spawn_radius_widens_when_too_tight_to_fit_everyone():
    static, _ = generate_static(random.Random(0), 64, 64, 0.2)
    hero, goal = (32, 32), (2, 2)

    with pytest.warns(UserWarning, match="widened"):
        enemies = place_enemies(
            random.Random(0), static, hero, goal, count=10, spawn_radius=1
        )

    assert len(enemies) == 10
    assert len(set(enemies)) == 10


def test_goal_reachability_is_recorded_not_rerolled():
    scenario = build_scenario(random.Random(0))
    reachable = scenario.grid.goal in scenario.grid.flood_reachable(scenario.hero)
    assert scenario.goal_reachable_at_init == reachable
