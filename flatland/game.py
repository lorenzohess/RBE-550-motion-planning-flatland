#!/usr/bin/env python3
"""Game state, the tick loop, and termination."""

import time
from dataclasses import dataclass, field
from enum import Enum

from .entities import MAX_TELEPORTS, Enemies, Enemy, Hero
from .grid import manhattan
from .observation import snapshot
from .rules import run_tick

DEFAULT_MAX_TICKS = 600


class Outcome(Enum):
    RUNNING = "running"
    WIN = "win"
    LOSS = "loss"
    TIMEOUT = "timeout"
    NO_PATH = "no_path"
    INTERRUPTED = "interrupted"

    @property
    def is_terminal(self):
        return self is not Outcome.RUNNING


@dataclass
class TickRecord:
    """One row of the per-tick log."""

    tick: int
    hero: tuple
    action: str
    enemies_alive: int
    wrecks: int
    dist_to_goal: int
    dist_nearest_enemy: object
    path_len: object
    replanned: bool
    nodes_expanded: object
    plan_ms: float
    teleports_left: int
    event: str


@dataclass
class RunResult:
    seed: object
    outcome: Outcome
    ticks: int
    cause: str = None
    path_len: int = 0
    enemies_destroyed: int = 0
    enemies_alive: int = 0
    teleports_used: int = 0
    total_plan_ms: float = 0.0
    rho_measured: float = 0.0
    goal_reachable_at_init: bool = True
    records: list = field(default_factory=list)


class Game:
    """Drives one run: hero acts, enemies act, check termination, repeat."""

    def __init__(
        self,
        scenario,
        planner,
        rng,
        max_ticks=DEFAULT_MAX_TICKS,
        adjacency_death=False,
        seed=None,
        teleport_budget=MAX_TELEPORTS,
    ):
        self.grid = scenario.grid
        self.hero = Hero(scenario.hero, max_teleports=teleport_budget)
        self.enemies = Enemies(
            [Enemy(cell, index) for index, cell in enumerate(scenario.enemies)]
        )
        self.scenario = scenario
        self.planner = planner
        self.rng = rng
        self.max_ticks = max_ticks
        self.adjacency_death = adjacency_death
        self.seed = seed

        self.tick_count = 0
        self.outcome = Outcome.RUNNING
        self.cause = None
        self.hero_trail = [self.hero.cell]
        self.records = []
        self.debug = None
        self.total_plan_ms = 0.0

    def observe(self):
        return snapshot(self.tick_count, self.grid, self.hero, self.enemies)

    def step(self):
        """Advance one tick. Returns the TickEvents, or None if already over."""
        if self.outcome.is_terminal:
            return None

        obs = self.observe()
        started = time.perf_counter()
        action = self.planner.act(obs)
        plan_ms = (time.perf_counter() - started) * 1000.0
        self.total_plan_ms += plan_ms
        self.debug = self.planner.debug()

        if action is None:
            # The planner concedes: no route exists and it declines to gamble on
            # the teleporter.
            self.outcome = Outcome.NO_PATH
            self.cause = "planner reported no path"
            self._record(action="CONCEDE", events=None, plan_ms=plan_ms)
            return None

        events = run_tick(
            self.grid,
            self.hero,
            self.enemies,
            action,
            self.rng,
            adjacency_death=self.adjacency_death,
        )
        self.tick_count += 1
        self.hero_trail.append(self.hero.cell)

        if events.reached_goal:
            self.outcome = Outcome.WIN
            self.cause = "hero reached the goal"
        elif events.hero_died:
            self.outcome = Outcome.LOSS
            self.cause = events.cause
        elif self.tick_count >= self.max_ticks:
            self.outcome = Outcome.TIMEOUT
            self.cause = f"tick cap {self.max_ticks} reached"

        self._record(action=str(action), events=events, plan_ms=plan_ms)
        return events

    def _record(self, action, events, plan_ms):
        debug = self.debug
        nearest = min(
            (manhattan(cell, self.hero.cell) for cell in self.enemies.cells),
            default=None,
        )
        notes = list(events.notes) if events is not None else []
        if events is not None and events.hero_died:
            notes.append("hero_died")
        if events is not None and events.reached_goal:
            notes.append("reached_goal")

        self.records.append(
            TickRecord(
                tick=self.tick_count,
                hero=self.hero.cell,
                action=action,
                enemies_alive=len(self.enemies.living),
                wrecks=int(self.grid.wreck_mask.sum()),
                dist_to_goal=manhattan(self.hero.cell, self.grid.goal),
                dist_nearest_enemy=nearest,
                path_len=len(debug.path) if debug and debug.path else None,
                replanned=bool(debug and debug.path),
                nodes_expanded=debug.nodes_expanded if debug else None,
                plan_ms=round(plan_ms, 4),
                teleports_left=self.hero.teleports_remaining,
                event=";".join(notes),
            )
        )

    def interrupt(self):
        if not self.outcome.is_terminal:
            self.outcome = Outcome.INTERRUPTED
            self.cause = "interrupted by user"

    def result(self):
        return RunResult(
            seed=self.seed,
            outcome=self.outcome,
            ticks=self.tick_count,
            cause=self.cause,
            path_len=len(self.hero_trail) - 1,
            enemies_destroyed=self.enemies.destroyed_count,
            enemies_alive=len(self.enemies.living),
            teleports_used=self.hero.teleports_used,
            total_plan_ms=round(self.total_plan_ms, 3),
            rho_measured=round(self.scenario.rho_measured, 5),
            goal_reachable_at_init=self.scenario.goal_reachable_at_init,
            records=self.records,
        )

    def run(self, on_tick=None):
        """Run to termination. ``on_tick(game, events)`` is called each tick."""
        try:
            while not self.outcome.is_terminal:
                events = self.step()
                if on_tick is not None:
                    on_tick(self, events)
        except KeyboardInterrupt:
            self.interrupt()
        return self.result()
