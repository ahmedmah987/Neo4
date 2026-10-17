"""Object-targeted exploration: in an open room the model agent should walk to the
objects it has not touched instead of testing every move at every position."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.graph_agent import GraphAgent
from neo4.model_agent import ModelAgent

WALL, AVATAR, GOAL = 3, 9, 14
DIRS = {GameAction.ACTION1: (1, 0), GameAction.ACTION2: (-1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)}


class OpenRoom:
    """16x16 cells of 4 px; border walls; three decorations and one goal object far apart."""

    objects = {(13, 2): 12, (2, 13): 6, (8, 8): 7, (13, 13): GOAL}

    def __init__(self):
        self.pos, self.state = (1, 1), GameState.NOT_PLAYED

    def _obs(self):
        o = FrameDataRaw(state=self.state, levels_completed=0, win_levels=1,
                         available_actions=[0, 1, 2, 3, 4])
        g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
        g[0:4, :] = g[60:64, :] = g[:, 0:4] = g[:, 60:64] = WALL
        for (cx, cy), col in self.objects.items():
            g[4 * cy : 4 * cy + 4, 4 * cx : 4 * cx + 4] = col
        x, y = self.pos
        g[4 * y : 4 * y + 4, 4 * x : 4 * x + 4] = AVATAR
        o.frame = [g]
        return o

    def reset(self):
        self.pos, self.state = (1, 1), GameState.NOT_FINISHED
        return self._obs()

    def step(self, action, data=None):
        if action is GameAction.RESET:
            return self.reset()
        if self.state is not GameState.NOT_FINISHED:
            return self._obs()
        dx, dy = DIRS.get(action, (0, 0))
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if nx in (0, 15) or ny in (0, 15):
            return self._obs()
        if (nx, ny) in self.objects:
            if self.objects[(nx, ny)] == GOAL:
                self.state = GameState.WIN
            return self._obs()  # decorations are solid
        self.pos = (nx, ny)
        return self._obs()


def run(agent, steps=3000):
    env = OpenRoom()
    obs = env.reset()
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.state == GameState.WIN:
            return used
    return steps


def test_model_agent_touches_objects_faster_than_exhaustive_exploration():
    model = [ModelAgent(seed=s, explore=True) for s in range(3)]
    m = [run(a) for a in model]
    g = [run(GraphAgent(seed=s)) for s in range(3)]
    assert sum(a.stats["explore_plans"] for a in model) >= 3
    assert all(len(a.dyn.touched) >= 1 for a in model)  # the goal itself ends the run before it is recorded
    assert sum(m) < sum(g), (m, g)
