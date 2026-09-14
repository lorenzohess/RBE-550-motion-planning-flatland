#!/usr/bin/env python3


from ..config import PlannerConfig
from ..entities import Action
from .base import Planner, PlannerDebug
from .enemy import time_to_reach
from ..grid import Grid, manhattan, Cell


class HeroPlanner(Planner):

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config if config is not None else PlannerConfig()
        # Set here rather than in act(): the name is read off the instance
        # handed to Game, and a planner that has not run a tick yet must still
        # name itself correctly.
        self.name = self.config.name
        self._debug = None

        variants = {
            "dijk": self.actDijk,
            "dijk-teleport": self.actDijkTeleport,
            "dijk-cost": self.actDijkCost,
        }
        if self.config.planner not in variants:
            raise ValueError(
                f"HeroPlanner cannot run variant {self.config.planner!r}; "
                f"choose one of {sorted(variants)}"
            )
        self._variant = variants[self.config.planner]

    def act(self, obs):
        return self._variant(obs)

    def debug(self):
        return self._debug

    def addEnemiesAsObstacle(self, obs):
        newGrid = Grid(obs.static.copy(), obs.goal)
        for enemy in obs.enemies:
            newGrid.add_obstacle(enemy)
        return newGrid

    def enemyCost(self, obs):
        """
        Returns cost(cell) which Dijk uses to compute cost to come to that cell.

        We model the cost as a function of the nearest enemy's Manhattan
        distance.  If the nearest enemy is within a certain radius, we compute
        cost by normalizing that distance and weighting it. If enemy outside,
        cost is 0.
        """
        if not obs.enemies:
            return None

        MIN_COST: float = 1.0
        enemyCostWeight: float = self.config.enemy_weight
        enemyRadius: int = self.config.enemy_radius

        def cost(cell):
            nearestEnemyDistance = min(manhattan(cell, enemy) for enemy in obs.enemies)
            normalizedEnemyDistance: float = (
                enemyRadius - nearestEnemyDistance
            ) / enemyRadius
            return MIN_COST + enemyCostWeight * max(0, normalizedEnemyDistance)

        return cost

    def returnActionFromPath(self, obs, path: list[Cell] | None) -> Action | None:
        if path is None:
            # If truly no path cause of obstacles, no enemy blocking
            pathWithoutEnemyPseudoObstacle = obs.grid.dijk(obs.hero, obs.goal)
            if pathWithoutEnemyPseudoObstacle is None:
                if obs.teleports_remaining <= 0:
                    self._debug = PlannerDebug(label="no path, teleports spent")
                    return None
                self._debug = PlannerDebug(label="no path, teleporting")
                return Action.teleport()

            # An enemy pseudo-obstacle was blocking path, wait til he moves
            self._debug = PlannerDebug(label="enemy blocking, waiting")
            return Action.wait()

        self._debug = PlannerDebug(path=path, label=f"d={len(path) - 1}")
        if len(path) < 2:
            return Action.wait()  # already standing on the goal

        return Action.move_to(path[1])  # or Action.teleport()

    def actDijk(self, obs):
        grid = self.addEnemiesAsObstacle(obs)
        path = grid.dijk(obs.hero, obs.goal, None)
        return self.returnActionFromPath(obs, path)

    def actDijkTeleport(self, obs):
        distanceToNearestEnemy: int | None = time_to_reach(obs.enemies, obs.hero)
        if (
            distanceToNearestEnemy is not None
            and distanceToNearestEnemy <= self.config.teleport_threshold
            and obs.teleports_remaining > 0
        ):
            return Action.teleport()

        return self.actDijk(obs)

    def actDijkCost(self, obs):
        newGrid = self.addEnemiesAsObstacle(obs)
        path = newGrid.dijk(obs.hero, obs.goal, self.enemyCost(obs))
        return self.returnActionFromPath(obs, path)
