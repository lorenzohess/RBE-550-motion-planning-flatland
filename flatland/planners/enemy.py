#!/usr/bin/env python3
"""The enemy policy, and prediction helpers built on it.

Enemies are governed by :func:`flatland.rules.enemy_intent`, which commits to one
randomly chosen distance-reducing step. This module exposes the same policy in a
form useful for *forecasting* it: because the tie-break is random, a prediction
cannot be a single trajectory, so these functions return the whole set of cells an
enemy could occupy.
"""

from collections.abc import Collection, Iterable, Sequence

from ..grid import DIRECTIONS, Cell, Grid, manhattan, step


def candidate_steps(enemy_cell, hero_cell):
    """Every step an enemy might take, i.e. all distance-reducing neighbours.

    Returns one cell when the enemy is axis-aligned with the hero, two when it is
    diagonal, and the enemy's own cell when it is already on the hero. Terrain is
    ignored, matching the enemies' blindness.
    """
    current = manhattan(enemy_cell, hero_cell)
    options = [
        candidate
        for direction in DIRECTIONS
        if manhattan(candidate := step(enemy_cell, direction), hero_cell) < current
    ]
    return options or [enemy_cell]


def predict_occupancy(
    grid: Grid,
    enemy_cells: Iterable[Cell],
    hero_path: Sequence[Cell],
    horizon: int,
) -> dict[int, set[Cell]]:
    """Forecast where enemies could be over the next ``horizon`` ticks.

    ``hero_path`` is the hero's assumed future, with ``hero_path[0]`` the current
    cell; if it is shorter than the horizon the last cell is held. It must be
    non-empty -- a ``Sequence`` because this both indexes and measures it. Returns
    ``{steps_ahead: set_of_cells}``, suitable for ``PlannerDebug.predicted_enemies``.
    Keys start at 1, and stop early if every branch has wrecked itself, so a
    caller must not assume the full ``1..horizon`` range is present.

    Branches that step into an obstacle are dropped, since such an enemy is
    destroyed. Enemy-enemy collisions are *not* modelled — they depend on the
    joint state of all enemies — so the forecast slightly overestimates how many
    survive, which is the safe direction to err in.
    """
    possible = {cell for cell in enemy_cells}
    forecast = {}
    for ahead in range(1, horizon + 1):
        hero_cell = hero_path[min(ahead, len(hero_path) - 1)]
        nxt = set()
        for cell in possible:
            for candidate in candidate_steps(cell, hero_cell):
                if candidate == hero_cell or not grid.is_obstacle(candidate):
                    nxt.add(candidate)
        forecast[ahead] = nxt
        possible = nxt
        if not possible:
            break
    return forecast


def time_to_reach(enemy_cells: Collection[Cell], hero_cell: Cell) -> int | None:
    """Ticks until the nearest enemy could contact the hero, ignoring terrain.

    A lower bound: obstacles can only slow enemies down or destroy them. Returns
    None when the roster is empty, which is a different thing from 0 -- callers
    that treat it as a number will read "all enemies dead" as "enemy on top of
    me". ``Collection`` rather than ``Iterable`` because the empty check needs a
    length; a generator would report truthy and then blow up in ``min``.
    """
    if not enemy_cells:
        return None
    return min(manhattan(cell, hero_cell) for cell in enemy_cells)
