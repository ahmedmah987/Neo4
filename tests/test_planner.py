import numpy as np
from arcengine import GameState

from neo4 import frame as fp
from neo4.model_agent import ModelAgent
from neo4.objects import extract
from neo4.planner import ahead, cells_ahead, plan_path
from tests.test_model_agent import GOAL, Room


def test_plan_path_finds_route_around_wall():
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[0:40, 20:22] = 3  # wall from the top, open at the bottom
    g[10:12, 5:7] = 9  # avatar 2x2
    g[10:12, 40:42] = 12  # goal
    me = next(o for o in extract(g) if o.color == 9)
    dirs = {1: (2, 0), 2: (-2, 0), 3: (0, -2), 4: (0, 2)}
    found = plan_path(g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols)
    assert found is not None
    actions, positions = found
    assert len(actions) == len(positions)
    # replay the plan on the grid: every step before the last enters only colour 0
    rel = me.pixels - np.array([me.y1, me.x1])
    pos = (me.x1, me.y1)
    for aid, nxt in zip(actions[:-1], positions[:-1]):
        assert cells_ahead(g, rel, pos, dirs[aid]) == {0}
        pos = nxt
    assert 12 in cells_ahead(g, rel, pos, dirs[actions[-1]])
    assert any(p[1] >= 40 for p in positions)  # the route went under the wall


def test_plan_path_returns_none_when_unreachable():
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[:, 20:22] = 3  # full-height wall
    g[10:12, 5:7] = 9
    g[10:12, 40:42] = 12
    me = next(o for o in extract(g) if o.color == 9)
    dirs = {1: (2, 0), 2: (-2, 0), 3: (0, -2), 4: (0, 2)}
    assert plan_path(g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols) is None


def test_model_agent_transfers_goal_to_next_level():
    agent = ModelAgent(seed=0)
    env = Room()
    obs = env.reset()
    solved_at = []
    for used in range(1, 3000 + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.levels_completed > len(solved_at):
            solved_at.append(used)
        if obs.state == GameState.WIN:
            break
    assert len(solved_at) == 2, solved_at
    first, second = solved_at[0], solved_at[1] - solved_at[0]
    assert GOAL in agent.dyn.goal_colors()
    assert agent.stats["plans"] >= 1 and agent.stats["plan_steps"] >= 10
    assert second < first / 4, (first, second)


def test_plan_path_sweeps_the_cells_between_jumps():
    """An avatar that jumps six cells must not plan through a thin wall that
    lies between two landing spots."""
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[0:40, 20:22] = 3  # wall from the top, open at the bottom; never a landing cell
    g[10:13, 2:5] = 9  # avatar 3x3 on a 6-cell lattice
    g[10:13, 44:47] = 12  # goal, straight ahead across the wall
    me = next(o for o in extract(g) if o.color == 9)
    dirs = {1: (6, 0), 2: (-6, 0), 3: (0, -6), 4: (0, 6)}
    found = plan_path(g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols)
    assert found is not None
    actions, positions = found
    rel = me.pixels - np.array([me.y1, me.x1])
    pos = (me.x1, me.y1)
    for aid, nxt in zip(actions[:-1], positions[:-1]):
        assert cells_ahead(g, rel, pos, dirs[aid]) == {0}  # no jump crosses the wall
        pos = nxt
    assert any(p[1] >= 40 for p in positions)  # the route went under the wall
    assert cells_ahead(g, rel, (14, 10), (6, 0)) == {0, 3}  # a jump over the wall sees it


def test_plan_path_respects_avoid_zone_and_forbidden_moves():
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[10:12, 4:6] = 9  # avatar 2x2
    g[10:12, 20:22] = 12  # goal straight ahead
    me = next(o for o in extract(g) if o.color == 9)
    dirs = {1: (2, 0), 2: (-2, 0), 3: (0, -2), 4: (0, 2)}
    direct = plan_path(g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols)
    assert direct is not None and all(a == 1 for a in direct[0])
    avoid = np.zeros((fp.GRID, fp.GRID), dtype=bool)
    avoid[8:14, 10:14] = True  # a patrol's reach across the direct route
    found = plan_path(g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols, avoid=avoid)
    assert found is not None and len(found[0]) > len(direct[0])
    assert all(not avoid[y, x] for _, (x, y) in zip(found[0], found[1]) for y in range(y, y + 2))
    fatal = {(4, 10, 1)}  # moving right from the start lost the game before
    found = plan_path(
        g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols,
        forbidden=lambda pos, aid: (pos[0], pos[1], aid) in fatal,
    )
    assert found is not None and found[0][0] != 1


def test_plan_path_waits_out_a_moving_hazard_by_detouring():
    """A patrol is predicted to cross the direct route at the moment the avatar
    would be there; the plan must take a longer way (there is no waiting move)."""
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[10:12, 4:6] = 9  # avatar 2x2
    g[10:12, 16:18] = 12  # goal: six moves to the right
    me = next(o for o in extract(g) if o.color == 9)
    dirs = {1: (2, 0), 2: (-2, 0), 3: (0, -2), 4: (0, 2)}
    direct = plan_path(g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols)
    assert direct is not None and len(direct[0]) == 6
    zones = [np.zeros((fp.GRID, fp.GRID), dtype=bool) for _ in range(6)]
    zones[1][8:14, 8:12] = True  # the patrol covers the route's third cell after two moves
    found = plan_path(g, me, dirs, lambda c: c == 0, lambda cols, coords: 12 in cols, avoid_t=zones)
    assert found is not None and len(found[0]) > 6
    # replay: at no time does the avatar enter the predicted zone
    rel = me.pixels - np.array([me.y1, me.x1])
    pos = (me.x1, me.y1)
    for t, (aid, nxt) in enumerate(zip(found[0], found[1])):
        _, coords = ahead(g, rel, pos, dirs[aid])
        if t < len(zones):
            assert not any(zones[t][y, x] for y, x in coords)
        pos = nxt
