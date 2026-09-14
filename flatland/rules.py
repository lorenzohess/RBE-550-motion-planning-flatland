#!/usr/bin/env python3
"""Turn resolution and collision outcomes.

One tick is: the hero acts, then every living enemy acts simultaneously. All
enemy outcomes are computed against the terrain as it stood at the *start* of the
enemy phase, so a wreck created this tick cannot affect another enemy until the
next tick.
"""

import warnings
from dataclasses import dataclass, field

from .entities import ActionKind
from .grid import manhattan
from .planners.enemy import candidate_steps


class IllegalHeroAction(UserWarning):
    """The hero's planner asked for something the rules forbid."""


@dataclass
class TickEvents:
    """What happened during one tick."""

    hero_died: bool = False
    reached_goal: bool = False
    cause: str = None
    destroyed: list = field(default_factory=list)
    wrecks: list = field(default_factory=list)
    illegal_action: bool = False
    teleported: bool = False
    notes: list = field(default_factory=list)


def _living_by_cell(enemies):
    return {enemy.cell: enemy for enemy in enemies.members if enemy.alive}


def teleport_destination(rng, grid, hero, enemies):
    """A uniformly random free cell, excluding enemies and the hero's own cell.

    Excluding the hero's cell matters: landing on yourself would silently consume
    one of only five activations and a tick, for no movement at all.
    """
    occupied = set(enemies.cells)
    occupied.add(hero.cell)
    candidates = [cell for cell in grid.free_cells() if cell not in occupied]
    if not candidates:
        return None
    return rng.choice(candidates)


def apply_hero_action(grid, hero, enemies, action, rng, events):
    """Apply the hero's action in place. Sets events.hero_died on contact.

    An illegal action warns and degrades to WAIT rather than raising: a planner
    under development should not lose the whole run to one bad tick, but the
    problem must not pass silently either.
    """
    if action.kind is ActionKind.MOVE:
        target = tuple(action.target)
        reason = None
        if not grid.in_bounds(target):
            reason = "out of bounds"
        elif manhattan(target, hero.cell) != 1:
            reason = "not 4-adjacent"
        elif grid.is_obstacle(target):
            reason = "obstacle"

        if reason is not None:
            warnings.warn(
                f"illegal hero move {hero.cell} -> {target} ({reason}); "
                f"treating as WAIT",
                IllegalHeroAction,
                stacklevel=2,
            )
            events.illegal_action = True
            events.notes.append(f"illegal_move:{reason}")
            return
        hero.cell = target

    elif action.kind is ActionKind.TELEPORT:
        if hero.teleports_remaining <= 0:
            warnings.warn(
                "teleport requested with none remaining; treating as WAIT",
                IllegalHeroAction,
                stacklevel=2,
            )
            events.illegal_action = True
            events.notes.append("illegal_teleport:exhausted")
            return
        destination = teleport_destination(rng, grid, hero, enemies)
        if destination is None:
            warnings.warn(
                "teleport requested but no free cell exists; treating as WAIT",
                IllegalHeroAction,
                stacklevel=2,
            )
            events.illegal_action = True
            events.notes.append("illegal_teleport:no_destination")
            return
        hero.cell = destination
        hero.teleports_used += 1
        events.teleported = True
        events.notes.append(f"teleport->{destination}")

    elif action.kind is not ActionKind.WAIT:
        raise ValueError(f"unknown action kind: {action.kind!r}")

    # The hero walked onto a living enemy.
    victim = _living_by_cell(enemies).get(hero.cell)
    if victim is not None:
        events.hero_died = True
        events.cause = "hero moved onto enemy"


def enemy_intent(grid, enemy_cell, hero_cell, rng):
    """The cell a blind enemy will step to: a random distance-reducing neighbour.

    Enemies perceive only the hero, so obstacles and other enemies are ignored
    here. Note that a step reducing Manhattan distance to an in-bounds hero is
    always itself in bounds, so this never returns an off-grid cell -- enemies
    cannot walk off the edge of the world.
    """
    # Always consult the rng, even for a single option, so that rng consumption
    # is one draw per enemy per tick regardless of geometry.
    return rng.choice(candidate_steps(enemy_cell, hero_cell))


def resolve_enemies(grid, hero, enemies, rng, events):
    """Move every living enemy simultaneously and apply collision outcomes."""
    living = enemies.living
    if not living:
        return

    intents = {enemy.index: enemy_intent(grid, enemy.cell, hero.cell, rng)
               for enemy in living}

    target_counts = {}
    for target in intents.values():
        target_counts[target] = target_counts.get(target, 0) + 1

    # Snapshot terrain so wrecks created this tick do not cascade within it.
    was_obstacle = {
        target: grid.is_obstacle(target)
        for target in target_counts
        if grid.in_bounds(target)
    }

    reached_hero = []
    new_wrecks = []
    for enemy in living:
        target = intents[enemy.index]

        if target == hero.cell:
            reached_hero.append(enemy.index)
            continue

        if not grid.in_bounds(target):
            # Unreachable in practice (see enemy_intent), kept as a guard.
            enemy.alive = False
            events.destroyed.append(enemy.index)
            new_wrecks.append(enemy.cell)
            events.notes.append(f"enemy{enemy.index}:boundary")
            continue

        if was_obstacle[target]:
            enemy.alive = False
            events.destroyed.append(enemy.index)
            new_wrecks.append(target)
            events.notes.append(f"enemy{enemy.index}:obstacle")
            continue

        if target_counts[target] > 1:
            enemy.alive = False
            events.destroyed.append(enemy.index)
            new_wrecks.append(target)
            events.notes.append(f"enemy{enemy.index}:collision")
            continue

        enemy.cell = target

    for cell in dict.fromkeys(new_wrecks):
        grid.add_obstacle(cell, wreck=True)
        events.wrecks.append(cell)

    if reached_hero:
        events.hero_died = True
        events.cause = f"enemy {reached_hero[0]} contacted hero"


def check_adjacency_death(hero, enemies, events):
    """Optional harder rule: a living enemy 4-adjacent to the hero is fatal."""
    if events.hero_died:
        return
    for enemy in enemies.living:
        if manhattan(enemy.cell, hero.cell) == 1:
            events.hero_died = True
            events.cause = f"enemy {enemy.index} adjacent to hero"
            return


def run_tick(grid, hero, enemies, action, rng, adjacency_death=False):
    """Resolve one full tick. Returns a TickEvents."""
    events = TickEvents()

    apply_hero_action(grid, hero, enemies, action, rng, events)
    if events.hero_died:
        return events

    # Arriving at the goal ends the run before the enemies get another move: the
    # hero got there first, so it is not killed on the tick it wins.
    if hero.cell == grid.goal:
        events.reached_goal = True
        return events

    if adjacency_death:
        check_adjacency_death(hero, enemies, events)
        if events.hero_died:
            return events

    resolve_enemies(grid, hero, enemies, rng, events)
    if not events.hero_died and adjacency_death:
        check_adjacency_death(hero, enemies, events)

    return events
