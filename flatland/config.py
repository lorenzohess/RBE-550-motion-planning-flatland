#!/usr/bin/env python3
"""Configuration objects for one sweep cell.

Separate from both ``cli`` and ``sweep`` so that a planner can read its knobs
from a config without importing either, which would be a cycle.
"""

from dataclasses import dataclass

PLANNER_IDS = ("baseline", "dijk", "dijk-teleport", "dijk-cost")


@dataclass(frozen=True)
class PlannerConfig:
    """A planner variant and its tuning knobs. Hashable, so it can key a cell."""

    planner: str = "dijk-cost"  # baseline | dijk | dijk-teleport | dijk-cost
    enemy_radius: int = 8  # dijk-cost only
    enemy_weight: float = 100.0  # dijk-cost only
    teleport_threshold: int = 2  # dijk-teleport only

    @property
    def name(self) -> str:
        """Stable id for the CSV, e.g. 'dijk-cost:r8:w100'.

        Only the knobs the variant actually reads go in the id, so a run is
        never labelled with a parameter that had no effect on it.
        """
        if self.planner == "dijk-cost":
            return f"dijk-cost:r{self.enemy_radius}:w{self.enemy_weight:g}"
        if self.planner == "dijk-teleport":
            return f"dijk-teleport:t{self.teleport_threshold}"
        return self.planner


@dataclass(frozen=True)
class WorldConfig:
    """The scenario a planner is tested against. Hashable, like PlannerConfig.

    ``teleport_budget`` is a rule rather than terrain, so it is handed to
    :class:`~flatland.game.Game` rather than baked into the scenario -- changing
    it must not change the map.
    """

    size: int = 64  # boards are square, so this sets both rows and cols
    enemy_spawn_radius: int | None = 20  # None = uniform over the whole map
    teleport_budget: int = 5
    enemy_count: int = 10
    rho: float = 0.2

    @property
    def max_ticks(self) -> int:
        """Tick cap, scaled to the board.

        The 600 default is sized for 64x64. On a 128x128 board a legitimate
        crossing plus detours can exceed it, and the run would record a
        ``timeout`` that is an artefact of the cap rather than a planner failure.
        """
        return 10 * self.size
