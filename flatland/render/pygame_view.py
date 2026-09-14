#!/usr/bin/env python3
"""pygame rendering: terrain, entities, HUD, and planner debug overlays."""

import numpy as np
import pygame

from ..game import Outcome
from ..grid import OBSTACLE

CELL_PX = 12
HUD_PX = 72
MARGIN = 8

FREE = (247, 247, 250)
GRIDLINE = (226, 226, 232)
# Terrain is deliberately low-contrast against the free cells: it is context,
# and the three saturated entities are what the eye should land on first.
OBSTACLE_COLOR = (168, 174, 188)
# Wrecks are the exception. They are dead enemies, so they borrow the enemy hue,
# darkened: against the light grey terrain they read by value as well as colour,
# and the kinship makes "this hulk used to be chasing me" legible at a glance.
WRECK_COLOR = (150, 58, 52)
GOAL_COLOR = (58, 178, 74)
HERO_COLOR = (38, 108, 220)
HERO_RING = (250, 250, 255)
ENEMY_COLOR = (203, 42, 42)
ENEMY_EDGE = (96, 10, 10)
TRAIL_COLOR = (110, 165, 230)
PATH_COLOR = (255, 130, 0)
VISITED_COLOR = (120, 170, 255, 70)
FRONTIER_COLOR = (255, 205, 60, 110)
PREDICT_COLOR = (220, 60, 60)
HUD_BG = (28, 30, 36)
HUD_TEXT = (236, 238, 244)
HUD_DIM = (150, 154, 166)

WIN_COLOR = (58, 178, 74)
LOSS_COLOR = (203, 42, 42)
NEUTRAL_COLOR = (200, 170, 60)

# Radar pings. The period is in rendered frames, so at the default 10 fps a ring
# takes about 1.4 s to sweep out. Staggering by enemy index keeps ten enemies
# from pulsing in lockstep, which reads as one blinking mass rather than ten.
PING_PERIOD = 14
PING_STAGGER = 3
PING_SPAN = 2.6          # how far the ring travels, in cells
PING_MAX_ALPHA = 210

# Overlay presets cycled with the `d` key.
OVERLAY_MODES = ("all", "path", "none")


class View:
    """Draws the game. Works headless under SDL_VIDEODRIVER=dummy."""

    def __init__(self, rows, cols, cell_px=CELL_PX, title="Flatland"):
        pygame.init()
        pygame.display.set_caption(title)

        self.rows = rows
        self.cols = cols
        self.cell = cell_px
        self.board_width = cols * cell_px
        self.board_height = rows * cell_px
        self.width = self.board_width + 2 * MARGIN
        self.height = self.board_height + 2 * MARGIN + HUD_PX

        self.surface = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        self.font_big = pygame.font.Font(None, 34)

        self.overlay_mode = "all"
        self.show_trail = True
        self.show_pings = True
        # Ping phase advances per rendered frame, not per tick, so the rings keep
        # sweeping while the game is paused -- which is exactly when you are
        # hunting for the entities on a cluttered board.
        self._frame = 0

    @property
    def size(self):
        return (self.width, self.height)

    # --- geometry -------------------------------------------------------------

    def _rect(self, cell):
        return pygame.Rect(
            MARGIN + cell[1] * self.cell,
            MARGIN + cell[0] * self.cell,
            self.cell,
            self.cell,
        )

    def _center(self, cell):
        return (
            MARGIN + cell[1] * self.cell + self.cell // 2,
            MARGIN + cell[0] * self.cell + self.cell // 2,
        )

    # --- layers ---------------------------------------------------------------

    def _draw_terrain(self, grid):
        self.surface.fill(HUD_BG)
        board = pygame.Rect(MARGIN, MARGIN, self.board_width, self.board_height)
        pygame.draw.rect(self.surface, FREE, board)

        if self.cell >= 8:
            for row in range(self.rows + 1):
                y = MARGIN + row * self.cell
                pygame.draw.line(self.surface, GRIDLINE,
                                 (MARGIN, y), (MARGIN + self.board_width, y))
            for col in range(self.cols + 1):
                x = MARGIN + col * self.cell
                pygame.draw.line(self.surface, GRIDLINE,
                                 (x, MARGIN), (x, MARGIN + self.board_height))

        obstacles = np.argwhere(grid.static == OBSTACLE)
        for row, col in obstacles:
            color = WRECK_COLOR if grid.wreck_mask[row, col] else OBSTACLE_COLOR
            pygame.draw.rect(self.surface, color, self._rect((row, col)))

    def _draw_cost_field(self, cost_field):
        finite = np.isfinite(cost_field)
        if not finite.any():
            return
        values = cost_field[finite]
        low, high = float(values.min()), float(values.max())
        if high <= low:
            return

        layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for row, col in np.argwhere(finite):
            fraction = (float(cost_field[row, col]) - low) / (high - low)
            # Cheap blue -> red ramp; avoids pulling matplotlib into the loop.
            color = (int(40 + 200 * fraction), 60, int(240 - 190 * fraction), 90)
            pygame.draw.rect(layer, color, self._rect((row, col)))
        self.surface.blit(layer, (0, 0))

    def _draw_cell_set(self, cells, color):
        if not cells:
            return
        layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for cell in cells:
            pygame.draw.rect(layer, color, self._rect(cell))
        self.surface.blit(layer, (0, 0))

    def _draw_predictions(self, predicted):
        if not predicted:
            return
        layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        horizon = max(predicted) or 1
        for ahead, cells in sorted(predicted.items()):
            # Nearer forecasts are drawn more strongly.
            alpha = max(18, int(120 * (1.0 - (ahead - 1) / max(horizon, 1))))
            inset = self.cell // 4
            for cell in cells:
                rect = self._rect(cell).inflate(-inset, -inset)
                pygame.draw.rect(layer, (*PREDICT_COLOR, alpha), rect, border_radius=2)
        self.surface.blit(layer, (0, 0))

    def _draw_trail(self, trail):
        if len(trail) < 2:
            return
        pygame.draw.lines(self.surface, TRAIL_COLOR, False,
                          [self._center(c) for c in trail], 2)

    def _draw_path(self, path):
        if not path or len(path) < 2:
            return
        points = [self._center(c) for c in path]
        pygame.draw.lines(self.surface, PATH_COLOR, False, points, 3)
        pygame.draw.circle(self.surface, PATH_COLOR, points[-1], max(3, self.cell // 3))

    def _draw_pings(self, hero, enemies):
        """Rings that expand outward from each entity and fade as they go."""
        layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Without this a ring near an edge would bleed into the margin and HUD.
        layer.set_clip(pygame.Rect(MARGIN, MARGIN,
                                   self.board_width, self.board_height))

        span = self.cell * PING_SPAN
        thickness = max(1, self.cell // 6)
        sources = [(0, hero.cell, HERO_COLOR)]
        sources += [(e.index + 1, e.cell, ENEMY_COLOR) for e in enemies]

        for offset, cell, color in sources:
            phase = ((self._frame + offset * PING_STAGGER) % PING_PERIOD) / PING_PERIOD
            # Start outside the marker itself, so the ring never covers it up.
            radius = int(self.cell * 0.6 + phase * span)
            alpha = int(PING_MAX_ALPHA * (1.0 - phase))
            if alpha <= 0:
                continue
            pygame.draw.circle(layer, (*color, alpha), self._center(cell),
                               radius, thickness)
        self.surface.blit(layer, (0, 0))

    def _draw_goal(self, goal):
        rect = self._rect(goal)
        pygame.draw.rect(self.surface, GOAL_COLOR, rect)
        pygame.draw.rect(self.surface, (255, 255, 255), rect, 2)

    def _draw_enemies(self, enemies):
        # The markers deliberately overflow their cells: at 12 px a shape inset
        # inside the cell is a speck, and overflow is what makes it a piece.
        grow = max(2, self.cell // 5)
        for enemy in enemies:
            rect = self._rect(enemy.cell).inflate(grow, grow)
            points = [
                (rect.centerx, rect.top),
                (rect.right, rect.bottom),
                (rect.left, rect.bottom),
            ]
            pygame.draw.polygon(self.surface, ENEMY_COLOR, points)
            pygame.draw.polygon(self.surface, ENEMY_EDGE, points,
                                max(1, self.cell // 8))

    def _draw_hero(self, hero):
        center = self._center(hero.cell)
        radius = max(5, int(self.cell * 0.62))
        pygame.draw.circle(self.surface, HERO_RING, center, radius)
        pygame.draw.circle(self.surface, HERO_COLOR, center,
                           max(3, radius - max(2, self.cell // 6)))

    # --- HUD ------------------------------------------------------------------

    def _draw_hud(self, game, paused):
        top = MARGIN * 2 + self.board_height
        pygame.draw.rect(self.surface, HUD_BG,
                         pygame.Rect(0, top, self.width, HUD_PX))

        debug = game.debug
        label = debug.label if debug and debug.label else game.planner.name
        nearest = min(
            (abs(c[0] - game.hero.cell[0]) + abs(c[1] - game.hero.cell[1])
             for c in game.enemies.cells),
            default=None,
        )

        left = (
            f"tick {game.tick_count}/{game.max_ticks}   "
            f"enemies {len(game.enemies.living)}/{len(game.enemies.members)}   "
            f"teleports {game.hero.teleports_remaining}   "
            f"goal d={abs(game.hero.cell[0] - game.grid.goal[0]) + abs(game.hero.cell[1] - game.grid.goal[1])}"
        )
        right = (
            f"planner: {label}"
            + (f"   nearest enemy {nearest}" if nearest is not None else "   all enemies down")
        )

        self.surface.blit(self.font.render(left, True, HUD_TEXT), (MARGIN, top + 8))
        self.surface.blit(self.font_small.render(right, True, HUD_DIM),
                          (MARGIN, top + 32))

        hint = (f"[space] pause  [.] step  [r] restart  "
                f"[d] overlays: {self.overlay_mode}  [t] trail  [p] pings  [esc] quit")
        self.surface.blit(self.font_small.render(hint, True, HUD_DIM),
                          (MARGIN, top + 50))

        if paused and not game.outcome.is_terminal:
            self._banner("PAUSED", NEUTRAL_COLOR)

        if game.outcome.is_terminal:
            colors = {
                Outcome.WIN: WIN_COLOR,
                Outcome.LOSS: LOSS_COLOR,
            }
            self._banner(game.outcome.value.upper().replace("_", " "),
                         colors.get(game.outcome, NEUTRAL_COLOR),
                         subtitle=game.cause)

    def _banner(self, text, color, subtitle=None):
        label = self.font_big.render(text, True, (255, 255, 255))
        pad = 14
        height = label.get_height() + 2 * pad
        if subtitle:
            sub = self.font_small.render(subtitle, True, (240, 240, 245))
            height += sub.get_height() + 4
        else:
            sub = None

        box = pygame.Surface((self.board_width, height), pygame.SRCALPHA)
        box.fill((*color, 225))
        y = MARGIN + (self.board_height - height) // 2
        self.surface.blit(box, (MARGIN, y))
        self.surface.blit(label, (MARGIN + (self.board_width - label.get_width()) // 2,
                                  y + pad))
        if sub is not None:
            self.surface.blit(sub, (MARGIN + (self.board_width - sub.get_width()) // 2,
                                    y + pad + label.get_height() + 4))

    # --- public ---------------------------------------------------------------

    def draw(self, game, paused=False):
        """Render the current game state to the surface."""
        debug = game.debug
        show = self.overlay_mode

        self._draw_terrain(game.grid)

        if show == "all" and debug is not None and debug.cost_field is not None:
            self._draw_cost_field(np.asarray(debug.cost_field, dtype=float))
        if show == "all" and debug is not None:
            self._draw_cell_set(debug.visited, VISITED_COLOR)
            self._draw_cell_set(debug.frontier, FRONTIER_COLOR)

        if self.show_trail and show != "none":
            self._draw_trail(game.hero_trail)
        if show == "all" and debug is not None:
            self._draw_predictions(debug.predicted_enemies)
        if show in ("all", "path") and debug is not None:
            self._draw_path(debug.path)

        # Pings answer "where is everyone", not "what is the planner thinking", so
        # they survive the overlay modes and have their own toggle.
        if self.show_pings:
            self._draw_pings(game.hero, game.enemies.living)

        # After the overlays: a planned path ends on the goal, and its endpoint
        # marker would otherwise hide the one cell the run is about.
        self._draw_goal(game.grid.goal)

        self._draw_enemies(game.enemies.living)
        self._draw_hero(game.hero)
        self._draw_hud(game, paused)

        self._frame += 1
        pygame.display.flip()

    def poll(self):
        """Drain the event queue. Returns a set of command strings."""
        commands = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                commands.add("quit")
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    commands.add("quit")
                elif event.key == pygame.K_SPACE:
                    commands.add("pause")
                elif event.key == pygame.K_PERIOD:
                    commands.add("step")
                elif event.key == pygame.K_r:
                    commands.add("restart")
                elif event.key == pygame.K_d:
                    self.overlay_mode = OVERLAY_MODES[
                        (OVERLAY_MODES.index(self.overlay_mode) + 1)
                        % len(OVERLAY_MODES)
                    ]
                elif event.key == pygame.K_t:
                    self.show_trail = not self.show_trail
                elif event.key == pygame.K_p:
                    self.show_pings = not self.show_pings
        return commands

    def close(self):
        pygame.quit()
