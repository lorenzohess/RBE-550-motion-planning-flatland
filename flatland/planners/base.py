#!/usr/bin/env python3
"""The planner contract, and the debug channel that feeds the visualizer."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class InvalidPlannerDebug(UserWarning):
    """A debug field the renderer cannot use. The field is dropped, not fatal."""


def _checked(name, value, types, expected):
    """Return ``value``, or ``None`` with a warning if it is not drawable.

    Only the container type is checked, not every cell: ``visited`` can hold
    thousands of entries and this runs once per tick. That is enough to catch the
    mistakes that would otherwise crash mid-run.
    """
    if value is None or isinstance(value, types):
        return value
    warnings.warn(
        f"PlannerDebug.{name} should be {expected}, got {value!r} -- dropping it. "
        "Note that `...` is the real object Ellipsis, not a placeholder; use None.",
        InvalidPlannerDebug,
        # _checked -> __post_init__ -> the dataclass's generated __init__ (which
        # reports as "<string>") -> the planner that built this. Only the last is
        # worth pointing at.
        stacklevel=4,
    )
    return None


@dataclass
class PlannerDebug:
    """Optional planner internals for rendering and logging.

    Every field is optional; the renderer draws whatever is present. Returning
    these costs nothing if you leave them unset, and ``cost_field`` plus ``path``
    are what make the "plots of the path planner state" figures in the report
    essentially free.

    Fields that cannot be drawn are dropped with a warning rather than raised on.
    This channel is instrumentation: a mistake in it should cost you an overlay,
    never a run that was otherwise going fine.
    """

    path: list = None
    """Planned route as a list of cells, drawn as a polyline."""

    predicted_enemies: dict = field(default_factory=dict)
    """``{steps_ahead: iterable_of_cells}``, drawn progressively fainter."""

    visited: set = None
    """Expanded nodes, drawn as a translucent wash."""

    frontier: set = None
    """Open set, drawn in a contrasting tint."""

    cost_field: object = None
    """A (rows, cols) float array drawn as a heatmap underlay, auto-normalized."""

    label: str = None
    """Free text shown in the HUD, e.g. the current strategy or mode."""

    nodes_expanded: int = None
    """Logged to the per-tick CSV for the report's search-effort plots."""

    def __post_init__(self):
        sets = (set, frozenset, list, tuple)
        self.path = _checked("path", self.path, (list, tuple),
                             "a list of (row, col) cells")
        self.visited = _checked("visited", self.visited, sets, "a set of cells")
        self.frontier = _checked("frontier", self.frontier, sets, "a set of cells")
        self.label = _checked("label", self.label, str, "a string")
        self.nodes_expanded = _checked("nodes_expanded", self.nodes_expanded, int,
                                       "an int")
        self.predicted_enemies = _checked(
            "predicted_enemies", self.predicted_enemies, dict,
            "a {steps_ahead: cells} dict",
        ) or {}
        # A cost field only has to survive np.asarray(..., dtype=float) and carry
        # a shape, so duck-type it rather than demanding an ndarray.
        if self.cost_field is not None and not hasattr(self.cost_field, "shape"):
            self.cost_field = _checked("cost_field", self.cost_field, (list, tuple),
                                       "a (rows, cols) float array")


class Planner(ABC):
    """Base class for the hero's brain.

    ``act`` is called once per tick and must return an
    :class:`~flatland.entities.Action`, or ``None`` to concede that no path
    exists. Conceding ends the run with outcome ``NO_PATH``; if you would rather
    gamble on the teleporter, return ``Action.teleport()`` instead, since
    teleporting is the only way to change which connected component you are in.

    Wrecks only ever *add* obstacles, so once the goal becomes unreachable it
    stays unreachable unless you teleport.
    """

    name = "planner"

    @abstractmethod
    def act(self, obs):
        """Return an Action for this tick, or None to concede."""

    def debug(self):
        """Optional PlannerDebug for the current tick."""
        return None
