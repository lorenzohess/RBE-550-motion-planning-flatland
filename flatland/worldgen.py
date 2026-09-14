#!/usr/bin/env python3
"""Adapter over the obstacleworld library, plus initial placement.

This is the only module that touches obstacleworld. It hands back a plain numpy
array in ``(row, col)`` orientation so the library's own ``get(x, y)`` convention
never leaks into the game.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from obstacleworld.obstacleworld.World import World

from .grid import FREE, Grid, manhattan

DEFAULT_ROWS = 64
DEFAULT_COLS = 64
DEFAULT_RHO = 0.2
DEFAULT_ENEMY_COUNT = 10

# Manhattan radius around the hero within which enemies spawn. Uniform placement
# over the whole map leaves enemies ~41 cells away, and ~95% of them wreck
# themselves on obstacles within ~18 ticks without ever threatening the hero.
# Pass None to recover that spec-literal uniform behaviour.
DEFAULT_ENEMY_SPAWN_RADIUS = 20


@dataclass
class Scenario:
    """A fully placed initial configuration."""

    grid: Grid
    hero: tuple
    enemies: list
    rho_measured: float
    goal_reachable_at_init: bool


def generate_static(rng, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, rho=DEFAULT_RHO):
    """Generate the tetromino obstacle field. Returns (array, measured_coverage)."""
    world = World(cols, rows, rng=rng)
    world.generateObstacleField(rho)
    # WorldGrid stores cells[y, x], which is already (row, col).
    return world.grid.cells.astype(np.uint8), world.coverage()


def _quadrant_cells(rows, cols, quadrant):
    """Cells belonging to a quadrant, given as (row_half, col_half) in {0,1}."""
    row_half, col_half = quadrant
    row_start, row_end = (0, rows // 2) if row_half == 0 else (rows // 2, rows)
    col_start, col_end = (0, cols // 2) if col_half == 0 else (cols // 2, cols)
    return row_start, row_end, col_start, col_end


def _free_in_quadrant(static, quadrant):
    rows, cols = static.shape
    row_start, row_end, col_start, col_end = _quadrant_cells(rows, cols, quadrant)
    window = static[row_start:row_end, col_start:col_end]
    free_rows, free_cols = np.nonzero(window == FREE)
    return list(zip((free_rows + row_start).tolist(), (free_cols + col_start).tolist()))


def place_hero_and_goal(rng, static):
    """Place the hero and goal in diagonally opposite quadrants.

    Guarantees a long journey (>= rows/2 + cols/2 Manhattan at minimum) and, as a
    side effect, that the hero never starts on the goal.
    """
    hero_quadrant = (rng.randint(0, 1), rng.randint(0, 1))
    goal_quadrant = (1 - hero_quadrant[0], 1 - hero_quadrant[1])

    hero_candidates = _free_in_quadrant(static, hero_quadrant)
    goal_candidates = _free_in_quadrant(static, goal_quadrant)
    if not hero_candidates or not goal_candidates:
        raise RuntimeError("no free cell available in a required quadrant")

    return rng.choice(hero_candidates), rng.choice(goal_candidates)


def place_enemies(rng, static, hero, goal, count=DEFAULT_ENEMY_COUNT,
                  spawn_radius=DEFAULT_ENEMY_SPAWN_RADIUS):
    """Place enemies on free cells, never adjacent to the hero.

    Excluding the hero's 4 neighbours means the hero always gets to move before
    it can be contacted, so no run is lost to a surprise attack on tick 0.
    """
    rows, cols = static.shape
    forbidden = {hero, goal}
    forbidden.update(
        (hero[0] + dr, hero[1] + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )

    free_rows, free_cols = np.nonzero(static == FREE)
    free = [
        cell
        for cell in zip(free_rows.tolist(), free_cols.tolist())
        if cell not in forbidden
    ]

    if spawn_radius is None:
        candidates = free
    else:
        radius = spawn_radius
        limit = rows + cols
        candidates = [c for c in free if manhattan(c, hero) <= radius]
        while len(candidates) < count and radius < limit:
            radius += 5
            candidates = [c for c in free if manhattan(c, hero) <= radius]
        if radius != spawn_radius:
            warnings.warn(
                f"enemy spawn radius widened {spawn_radius} -> {radius} "
                f"to fit {count} enemies",
                stacklevel=2,
            )

    if len(candidates) < count:
        raise RuntimeError(
            f"only {len(candidates)} spawnable cells for {count} enemies"
        )
    return rng.sample(candidates, count)


def build_scenario(rng, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, rho=DEFAULT_RHO,
                   enemy_count=DEFAULT_ENEMY_COUNT,
                   enemy_spawn_radius=DEFAULT_ENEMY_SPAWN_RADIUS):
    """Generate a world and place the hero, goal and enemies.

    An unreachable goal is recorded rather than re-rolled: the assignment
    requires the planner to handle the no-path case, and at rho=0.2 a genuinely
    unsolvable pair arises under 1% of the time, so re-rolling would hide it.
    """
    static, rho_measured = generate_static(rng, rows, cols, rho)
    hero, goal = place_hero_and_goal(rng, static)
    grid = Grid(static, goal)
    enemies = place_enemies(rng, static, hero, goal, enemy_count, enemy_spawn_radius)

    return Scenario(
        grid=grid,
        hero=hero,
        enemies=enemies,
        rho_measured=rho_measured,
        goal_reachable_at_init=goal in grid.flood_reachable(hero),
    )
