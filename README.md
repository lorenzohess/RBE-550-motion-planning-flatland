# HW2 Flatland

## Setup

This repo uses a submodule for the tetromino world generator. Clone with:

    git clone --recurse-submodules https://github.com/lorenzohess/<repo>.git

If you already cloned without it:

    git submodule update --init
    
Then: `.venv/bin/pip install -r requirements.txt`.

## Running

```bash
python main.py --planner baseline --seed 1                  # headless, writes CSV logs
python main.py --planner hero --seed 1                      # use my hero planner
python main.py --planner hero --seed 1 --live               # watch it in a window
python main.py --planner hero --seed 1 --record run.mp4     # write an animation
python main.py --planner hero --sweep 200                   # 200 seeds, summarized
```

Keys in `--live`: `space` pause, `.` single-step, `r` restart the same seed,
`d` cycle overlays, `t` toggle the hero's trail, `p` toggle the radar pings,
`esc` quit. Single-step plus overlays is the fastest way to debug a planner.

## Layout

```
flatland/
  grid.py         Grid, GridView, graph primitives
  worldgen.py     obstacleworld adapter + placement
  entities.py     Hero, Enemy, Action
  rules.py        turn resolution and collisions
  game.py         tick loop, outcomes
  observation.py  the planner's snapshot
  planners/       base.py, enemy.py, baseline.py, hero.py (yours)
  render/         pygame_view.py, recorder.py
  instrument.py   CSV logging
  plots.py        report figures
  cli.py          argument parsing
```
