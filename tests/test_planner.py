import numpy as np
from arcengine import GameState

from neo4 import frame as fp
from neo4.model_agent import ModelAgent
from neo4.objects import extract
from neo4.planner import cells_ahead, plan_path
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
