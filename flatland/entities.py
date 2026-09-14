#!/usr/bin/env python3
"""Hero and enemy state, and the hero's action space."""

from dataclasses import dataclass, field
from enum import Enum, auto

MAX_TELEPORTS = 5


class ActionKind(Enum):
    MOVE = auto()
    WAIT = auto()
    TELEPORT = auto()


@dataclass(frozen=True)
class Action:
    """What the hero does on one tick.

    MOVE carries an absolute destination cell rather than a direction, since a
    planner usually has a path in hand and ``Action.move_to(path[1])`` is the
    natural call. Use ``grid.step(cell, NORTH)`` to build one from a direction.
    """

    kind: ActionKind
    target: tuple = None

    @classmethod
    def move_to(cls, cell):
        return cls(ActionKind.MOVE, tuple(cell))

    @classmethod
    def wait(cls):
        return cls(ActionKind.WAIT)

    @classmethod
    def teleport(cls):
        return cls(ActionKind.TELEPORT)

    def __str__(self):
        if self.kind is ActionKind.MOVE:
            return f"MOVE{self.target}"
        return self.kind.name


@dataclass
class Hero:
    cell: tuple
    teleports_used: int = 0
    max_teleports: int = MAX_TELEPORTS

    @property
    def teleports_remaining(self):
        return self.max_teleports - self.teleports_used


@dataclass
class Enemy:
    cell: tuple
    index: int
    alive: bool = True


@dataclass
class Enemies:
    """The enemy roster. Wrecks are removed from play but keep their index."""

    members: list = field(default_factory=list)

    @property
    def living(self):
        return [enemy for enemy in self.members if enemy.alive]

    @property
    def cells(self):
        return tuple(enemy.cell for enemy in self.members if enemy.alive)

    @property
    def destroyed_count(self):
        return sum(1 for enemy in self.members if not enemy.alive)
