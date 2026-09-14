"""Rendering and recording smoke tests. All run headless."""

import os
import random
import shutil
import subprocess

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from flatland.game import Game  # noqa: E402
from flatland.planners.baseline import BaselinePlanner  # noqa: E402
from flatland.render.pygame_view import View  # noqa: E402
from flatland.render.recorder import Recorder  # noqa: E402
from flatland.worldgen import build_scenario  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@pytest.fixture
def game():
    rng = random.Random(1)
    return Game(build_scenario(rng), BaselinePlanner(), rng, seed=1)


@pytest.fixture
def view():
    v = View(64, 64, cell_px=6)
    yield v
    v.close()


def test_view_size_accounts_for_board_margins_and_hud(view):
    width, height = view.size
    assert width == 64 * 6 + 16
    assert height > 64 * 6 + 16  # HUD strip


def test_draw_works_before_and_after_the_planner_has_run(view, game):
    view.draw(game)  # no debug yet
    game.step()
    view.draw(game)  # debug now populated with a path and predictions
    assert game.debug is not None and game.debug.path


def test_draw_survives_every_overlay_mode(view, game):
    game.step()
    for mode in ("all", "path", "none"):
        view.overlay_mode = mode
        view.draw(game)


def test_draw_handles_a_cost_field_and_search_sets(view, game):
    import numpy as np

    game.step()
    game.debug.cost_field = np.random.default_rng(0).random((64, 64))
    game.debug.visited = {(r, c) for r in range(10) for c in range(10)}
    game.debug.frontier = {(10, c) for c in range(10)}
    view.draw(game)


def test_draw_handles_a_degenerate_cost_field(view, game):
    import numpy as np

    game.step()
    game.debug.cost_field = np.full((64, 64), np.inf)
    view.draw(game)
    game.debug.cost_field = np.zeros((64, 64))  # no spread to normalize
    view.draw(game)


def test_pings_animate_on_rendered_frames_not_on_ticks(view, game):
    """The rings must keep sweeping while the game is paused."""
    view.draw(game)
    view.draw(game)
    assert view._frame == 2

    before = pygame.image.tobytes(view.surface, "RGB")
    view.draw(game)  # same game state, later in the ping cycle
    assert pygame.image.tobytes(view.surface, "RGB") != before


def test_pings_can_be_switched_off(view, game):
    view.draw(game)
    with_pings = pygame.image.tobytes(view.surface, "RGB")

    view.show_pings = False
    view._frame = 0
    view.draw(game)
    assert pygame.image.tobytes(view.surface, "RGB") != with_pings


def test_pings_survive_every_overlay_mode_and_a_wiped_out_roster(view, game):
    """Pings are entity locators, not planner debug, so 'none' keeps them."""
    for mode in ("all", "path", "none"):
        view.overlay_mode = mode
        view.draw(game)

    for enemy in game.enemies.members:
        enemy.alive = False
    view.draw(game)  # hero-only ping, no enemies to stagger against


def test_draw_renders_the_terminal_banner(view, game):
    game.run()
    assert game.outcome.is_terminal
    view.draw(game)


@needs_ffmpeg
def test_recorder_writes_a_playable_mp4(tmp_path, view, game):
    out = tmp_path / "run.mp4"
    with Recorder(out, view.size, fps=10) as recorder:
        for _ in range(5):
            view.draw(game)
            recorder.add(view.surface)
            game.step()
        frames_before_hold = recorder.frames

    assert out.stat().st_size > 0
    assert frames_before_hold == 5

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, check=True,
    )
    codec, width, height = probe.stdout.split()
    assert codec == "h264"
    assert (int(width), int(height)) == view.size


@needs_ffmpeg
def test_recorder_creates_missing_output_directories(tmp_path, view, game):
    """ffmpeg fails at open time, which would surface only at close()."""
    out = tmp_path / "does" / "not" / "exist" / "run.mp4"
    with Recorder(out, view.size, fps=10) as recorder:
        view.draw(game)
        recorder.add(view.surface)

    assert out.stat().st_size > 0


@needs_ffmpeg
def test_recorder_rejects_a_mismatched_frame_size(tmp_path, view, game):
    recorder = Recorder(tmp_path / "bad.mp4", view.size, fps=10)
    try:
        with pytest.raises(ValueError, match="frame size"):
            recorder.add(pygame.Surface((10, 10)))
    finally:
        recorder.close()


@needs_ffmpeg
def test_recorder_close_is_idempotent(tmp_path, view, game):
    recorder = Recorder(tmp_path / "once.mp4", view.size, fps=10)
    view.draw(game)
    recorder.add(view.surface)
    recorder.close()
    recorder.close()  # must not raise


@needs_ffmpeg
def test_hold_repeats_the_final_frame(tmp_path, view, game):
    recorder = Recorder(tmp_path / "hold.mp4", view.size, fps=10)
    view.draw(game)
    recorder.add(view.surface)
    recorder.hold(seconds=1.0)
    recorder.close()

    assert recorder.frames == 11  # one real frame plus 10 held
