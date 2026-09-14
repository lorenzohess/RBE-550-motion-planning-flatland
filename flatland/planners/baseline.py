#!/usr/bin/env python3
"""A deliberately naive baseline planner.

This exists so the game, renderer and instrumentation can be exercised before the
real hero planner lands, and so the report has something to compare against. It
runs BFS to the goal and ignores the enemies completely, which makes it a useful
control: any improvement your planner shows over it is attributable to reasoning
about the enemies rather than to pathfinding.
"""

from ..entities import Action
from .base import Planner, PlannerDebug
from .enemy import predict_occupancy


class BaselinePlanner(Planner):
    """Shortest path to the goal, enemy-blind. Replans every tick."""

    name = "baseline-bfs"

    def __init__(self, predict_horizon=6):
        self.predict_horizon = predict_horizon
        self._debug = None

    def act(self, obs):
        path = obs.grid.bfs(obs.hero, obs.goal)

        if path is None:
            self._debug = PlannerDebug(label="no path", path=None)
            return None

        # Visited set is not tracked by Grid.bfs, so report what we can: the
        # route, and where the enemies might be if we follow it.
        self._debug = PlannerDebug(
            path=path,
            label=f"bfs d={len(path) - 1}",
            nodes_expanded=None,
            predicted_enemies=predict_occupancy(
                obs.grid, obs.enemies, path, self.predict_horizon
            ),
        )

        if len(path) < 2:
            return Action.wait()
        return Action.move_to(path[1])

    def debug(self):
        return self._debug
