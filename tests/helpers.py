"""Shared test scaffolding."""

import numpy as np

from flatland.entities import Enemies, Enemy, Hero
from flatland.grid import OBSTACLE, Grid


class ScriptedRNG:
    """A stand-in for random.Random whose choices are scripted.

    Enemy movement picks randomly among distance-reducing steps, so tests need to
    pin down which one each enemy takes. ``picks`` is consumed in call order;
    once exhausted, ``choice`` falls back to the first option.
    """

    def __init__(self, picks=()):
        self.picks = [tuple(p) for p in picks]
        self.calls = []

    def choice(self, seq):
        seq = list(seq)
        self.calls.append(seq)
        if self.picks:
            wanted = self.picks.pop(0)
            assert wanted in seq, f"scripted pick {wanted} not among options {seq}"
            return wanted
        return seq[0]

    def sample(self, population, k):
        return list(population)[:k]

    def randint(self, a, b):
        return a


def make_world(rows, cols, obstacles=(), goal=None, hero=(0, 0), enemies=()):
    """Build a hand-specified Grid/Hero/Enemies triple."""
    static = np.zeros((rows, cols), dtype=np.uint8)
    for cell in obstacles:
        static[cell[0], cell[1]] = OBSTACLE
    grid = Grid(static, goal if goal is not None else (rows - 1, cols - 1))
    roster = Enemies([Enemy(tuple(cell), index) for index, cell in enumerate(enemies)])
    return grid, Hero(tuple(hero)), roster
