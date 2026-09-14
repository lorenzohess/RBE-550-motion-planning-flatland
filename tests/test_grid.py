"""Graph primitives, including the no-path case."""

from flatland.grid import EAST, NORTH, Grid, manhattan, step
from helpers import make_world


def test_manhattan_and_step():
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((2, 2), (2, 2)) == 0
    assert step((1, 1), NORTH) == (0, 1)
    assert step((1, 1), EAST) == (1, 2)


def test_neighbors_excludes_obstacles_and_out_of_bounds():
    grid, _, _ = make_world(3, 3, obstacles=[(0, 1)])
    # Corner (0,0): (0,1) is an obstacle, (-1,0) and (0,-1) are off-grid.
    assert grid.neighbors((0, 0)) == [(1, 0)]


def test_neighbors_all_includes_obstacles_but_not_out_of_bounds():
    grid, _, _ = make_world(3, 3, obstacles=[(0, 1)])
    assert sorted(grid.neighbors_all((0, 0))) == [(0, 1), (1, 0)]


def test_bfs_returns_shortest_path_inclusive_of_endpoints():
    grid, _, _ = make_world(5, 5)
    path = grid.bfs((0, 0), (0, 3))
    assert path[0] == (0, 0)
    assert path[-1] == (0, 3)
    assert len(path) == 4
    assert all(manhattan(a, b) == 1 for a, b in zip(path, path[1:]))


def test_bfs_routes_around_an_obstacle():
    grid, _, _ = make_world(5, 5, obstacles=[(0, 1), (1, 1)])
    path = grid.bfs((0, 0), (0, 2))
    assert path is not None
    assert all(not grid.is_obstacle(cell) for cell in path)
    # Column 1 is blocked at rows 0-1, so the path must dip to (2,1): three
    # steps down-and-over, three back up-and-over, giving 7 cells.
    assert len(path) == 7


def test_bfs_start_equals_goal():
    grid, _, _ = make_world(3, 3)
    assert grid.bfs((1, 1), (1, 1)) == [(1, 1)]


def test_bfs_returns_none_when_endpoint_is_an_obstacle():
    grid, _, _ = make_world(3, 3, obstacles=[(2, 2)])
    assert grid.bfs((0, 0), (2, 2)) is None
    assert grid.bfs((2, 2), (0, 0)) is None


def test_walled_off_goal_has_no_path():
    """Hand-built unsolvable fixture.

    Random maps at rho=0.2 leave a start/goal pair unreachable under 1% of the
    time, so the NO_PATH outcome needs an explicit scenario like this one.
    """
    #  . . . . .
    #  . . # . .
    #  . . # . .     goal at (2,4) sealed behind a wall
    #  . . # . .
    #  . . # . .
    obstacles = [(row, 2) for row in range(5)]
    grid, _, _ = make_world(5, 5, obstacles=obstacles, goal=(2, 4))

    assert grid.bfs((2, 0), (2, 4)) is None
    reachable = grid.flood_reachable((2, 0))
    assert (2, 4) not in reachable
    assert len(reachable) == 10  # the two free columns left of the wall


def test_flood_reachable_includes_start_and_excludes_obstacles():
    grid, _, _ = make_world(3, 3, obstacles=[(1, 1)])
    reachable = grid.flood_reachable((0, 0))
    assert (0, 0) in reachable
    assert (1, 1) not in reachable
    assert len(reachable) == 8


def test_flood_reachable_from_an_obstacle_is_empty():
    grid, _, _ = make_world(3, 3, obstacles=[(1, 1)])
    assert grid.flood_reachable((1, 1)) == set()


def test_add_obstacle_marks_wreck_mask_only_when_requested():
    grid, _, _ = make_world(3, 3)
    grid.add_obstacle((0, 0), wreck=True)
    grid.add_obstacle((0, 1), wreck=False)

    assert grid.is_obstacle((0, 0)) and grid.wreck_mask[0, 0]
    assert grid.is_obstacle((0, 1)) and not grid.wreck_mask[0, 1]


def test_free_cells_reflects_added_obstacles():
    grid, _, _ = make_world(2, 2)
    assert len(grid.free_cells()) == 4
    grid.add_obstacle((0, 0))
    assert sorted(grid.free_cells()) == [(0, 1), (1, 0), (1, 1)]
