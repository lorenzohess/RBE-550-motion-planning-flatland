#!/usr/bin/env python3
"""The read-only snapshot handed to a planner each tick."""

from dataclasses import dataclass

from .grid import GridView


@dataclass(frozen=True)
class Observation:
    """Everything the hero perceives — which, being omniscient, is everything.

    ``grid`` is a query-only view: ``neighbors``, ``passable``, ``bfs``,
    ``flood_reachable`` and friends all work, but mutation raises. ``static`` is
    the same data as a non-writeable numpy array if you would rather work with it
    directly (0 = free, 1 = obstacle, wrecks included).
    """

    tick: int
    grid: GridView
    hero: tuple
    goal: tuple
    enemies: tuple
    teleports_remaining: int

    @property
    def static(self):
        return self.grid.static

    @property
    def rows(self):
        return self.grid.rows

    @property
    def cols(self):
        return self.grid.cols


def snapshot(tick, grid, hero, enemies):
    """Build an Observation from live game state."""
    return Observation(
        tick=tick,
        grid=GridView(grid),
        hero=hero.cell,
        goal=grid.goal,
        enemies=enemies.cells,
        teleports_remaining=hero.teleports_remaining,
    )
