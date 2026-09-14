#!/usr/bin/env python3
"""Static grid and graph primitives.

Coordinate convention: a cell is ``(row, col)``, indexable directly into the numpy
array. Row 0 renders at the top of the window. The names ``x`` and ``y`` are
deliberately avoided throughout ``flatland`` to keep this unambiguous.
"""

from collections import deque
from collections.abc import Callable

import numpy as np
from math import inf
import heapq

# A cell is always (row, col). The alias is here to make the search signatures
# readable; the rest of the module predates it and is deliberately unannotated.
Cell = tuple[int, int]

# Price of *entering* a cell. Must be strictly positive, or Dijkstra's
# settle-once guarantee does not hold.
CostFn = Callable[[Cell], float]

FREE = 0
OBSTACLE = 1

NORTH = (-1, 0)
SOUTH = (1, 0)
WEST = (0, -1)
EAST = (0, 1)

# 4-connected, per the assignment configuration.
DIRECTIONS = (NORTH, SOUTH, WEST, EAST)


def manhattan(a: Cell, b: Cell) -> int:
    """4-connected distance between two cells, ignoring obstacles."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step(cell, direction):
    """The cell reached by moving one step in a direction. No validity check."""
    return (cell[0] + direction[0], cell[1] + direction[1])


class Grid:
    """Terrain plus the goal. Entities live in Game, not here.

    ``static`` holds terrain only, so a hero standing on the goal is never
    ambiguous. Destroyed enemies become ordinary obstacles; ``wreck_mask``
    records which ones came from wrecks purely so the renderer can tint them.
    """

    def __init__(self, static, goal):
        self.static = static
        self.rows, self.cols = static.shape
        self.goal = goal
        self.wreck_mask = np.zeros(static.shape, dtype=bool)

    def in_bounds(self, cell):
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_obstacle(self, cell):
        return self.static[cell[0], cell[1]] == OBSTACLE

    def passable(self, cell: Cell) -> bool:
        """In bounds and not an obstacle."""
        return self.in_bounds(cell) and not self.is_obstacle(cell)

    def neighbors(self, cell: Cell) -> list[Cell]:
        """4-connected neighbors that are in bounds and not obstacles."""
        result = []
        for direction in DIRECTIONS:
            candidate = step(cell, direction)
            if self.passable(candidate):
                result.append(candidate)
        return result

    def neighbors_all(self, cell):
        """4-connected neighbors that are in bounds, obstacles included.

        Enemies are blind to obstacles, so their intent is computed with this.
        """
        return [
            c for direction in DIRECTIONS if self.in_bounds(c := step(cell, direction))
        ]

    def add_obstacle(self, cell, wreck=False):
        """Turn a cell into an obstacle. Used when an enemy is destroyed."""
        self.static[cell[0], cell[1]] = OBSTACLE
        if wreck:
            self.wreck_mask[cell[0], cell[1]] = True

    def free_cells(self):
        rows, cols = np.nonzero(self.static == FREE)
        return list(zip(rows.tolist(), cols.tolist()))

    def bfs(self, start, goal):
        """Shortest 4-connected path avoiding obstacles, or None.

        Returns the full path including both endpoints.
        """
        if not self.passable(start) or not self.passable(goal):
            return None
        if start == goal:
            return [start]

        came_from = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.neighbors(current):
                if neighbor in came_from:
                    continue
                came_from[neighbor] = current
                if neighbor == goal:
                    return self._reconstruct(came_from, goal)
                queue.append(neighbor)
        return None

    def dijk(
        self,
        start: Cell,
        goal: Cell,
        cost: CostFn | None = None,
    ) -> list[Cell] | None:
        """Cheapest 4-connected path from start to goal, or None.

        ``cost(cell)`` is the price of entering ``cell`` and must be positive.
        Leave it None for unit cost, which makes this equivalent to :meth:`bfs`
        -- useful as a self-check, useless otherwise. The point of this method is
        a non-uniform cost, e.g. one that prices proximity to enemies.

        Returns the full path including both endpoints.
        """
        if not self.passable(start) or not self.passable(goal):
            return None
        if start == goal:
            return [start]

        # Cell is a tuple, so it can be used to index a dictionary map.
        costToComeToCells: dict[Cell, float] = {start: 0.0}
        # This dict maps a cell to the cell from it was arrived at. This data
        # structure is used by _reconstruct to report the path from start to
        # goal.
        arrivedFromCells: dict[Cell, Cell | None] = {start: None}
        # Store the cells to visit in a list that will be called as a priority
        # queue.  The start cell is, by definition, the initial cell in the
        # queue with, by definition, cost 0.
        cellsToVisitPrQ: list[tuple[float, Cell]] = [(0.0, start)]

        while cellsToVisitPrQ:
            currentlyExpandedCellCost, currentlyExpandedCell = heapq.heappop(
                cellsToVisitPrQ
            )

            # We push duplicates onto the queue if they have lower cost, rather
            # than modifying existing nodes on the queue or deleting an
            # re-inserting. This check ensures that we skip the higher-cost
            # duplicate if it gets to the top of the queue.
            if currentlyExpandedCellCost > costToComeToCells[currentlyExpandedCell]:
                continue

            # If we're at goal, reconstruct the path so far. For Dijk to give
            # cheapest path, it requires that nextCell be popped from the heap,
            # not just connected to the current cell.
            if currentlyExpandedCell == goal:
                return self._reconstruct(arrivedFromCells, goal)

            for neighborCell in self.neighbors(currentlyExpandedCell):
                travelCost = 1.0 if cost is None else cost(neighborCell)
                costToNeighborCell = currentlyExpandedCellCost + travelCost

                if costToNeighborCell < costToComeToCells.get(neighborCell, inf):
                    # If the cost we just computed it less than what we computed in
                    # a previous Dijk iteration, update the costToComeToCell mapping.
                    costToComeToCells[neighborCell] = costToNeighborCell
                    # Then, store that we arrived to that neighbor from the current cell.
                    arrivedFromCells[neighborCell] = currentlyExpandedCell
                    # Then, push the duplicate, lower-cost neighbor onto the
                    # queue of cells to continue exploring.
                    heapq.heappush(cellsToVisitPrQ, (costToNeighborCell, neighborCell))

            # Somehow we didn't find the goal, so return None.
        return None

    @staticmethod
    def _reconstruct(came_from, goal):
        path = [goal]
        while came_from[path[-1]] is not None:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def flood_reachable(self, start):
        """Every passable cell reachable from start, including start itself."""
        if not self.passable(start):
            return set()
        seen = {start}
        queue = deque([start])
        while queue:
            for neighbor in self.neighbors(queue.popleft()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return seen


class GridView(Grid):
    """Query-only view of a Grid, handed to planners.

    Shares the underlying array rather than copying it, but marks it
    non-writeable and refuses mutation, so a planner cannot alter the world it is
    reasoning about.
    """

    def __init__(self, grid):
        self.static = grid.static.view()
        self.static.flags.writeable = False
        self.rows = grid.rows
        self.cols = grid.cols
        self.goal = grid.goal

    def add_obstacle(self, cell, wreck=False):
        raise TypeError("the observation's grid is read-only")
