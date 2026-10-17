"""A 16x16-cell room (4 px per cell): walls, lava, a goal. The avatar moves one
cell per keyboard action. Reaching the goal completes the level; touching lava
loses the game. Two levels with the goal in different corners."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.graph_agent import GraphAgent
from neo4.model_agent import ModelAgent

WALL, LAVA, GOAL, AVATAR = 3, 11, 12, 9
DIRS = {GameAction.ACTION1: (1, 0), GameAction.ACTION2: (-1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)}


class Room:
    goals = [(14, 14), (1, 14)]

    def __init__(self):
        self.level, self.state = 0, GameState.NOT_PLAYED
        self.cells = np.zeros((16, 16), dtype=np.int8)
        self.cells[0, :] = self.cells[-1, :] = self.cells[:, 0] = self.cells[:, -1] = WALL
        self.cells[3:13, 8] = LAVA
        self.cells[8, 8] = 0  # a gap in the lava wall
        self.pos = (1, 1)

    def _obs(self):
        o = FrameDataRaw(state=self.state, levels_completed=self.level, win_levels=2,
                         available_actions=[0, 1, 2, 3, 4])
        if self.state is not GameState.GAME_OVER:
            g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
            for (cy, cx), v in np.ndenumerate(self.cells):
                g[4 * cy : 4 * cy + 4, 4 * cx : 4 * cx + 4] = v
            gx, gy = self.goals[min(self.level, 1)]
            g[4 * gy : 4 * gy + 4, 4 * gx : 4 * gx + 4] = GOAL
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
        cell = self.cells[ny, nx]
        if cell == WALL:
            return self._obs()
        if cell == LAVA:
            self.state = GameState.GAME_OVER
            return self._obs()
        self.pos = (nx, ny)
        if self.pos == self.goals[min(self.level, 1)]:
            self.level += 1
            self.pos = (1, 1)
            if self.level == 2:
                self.state = GameState.WIN
        return self._obs()


def run(agent, env, steps):
    obs = env.reset()
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.state == GameState.WIN:
            return used
    return steps


def test_model_agent_learns_rules_and_solves_room():
    agent = ModelAgent(seed=0)
    used = run(agent, Room(), 3000)
    dyn = agent.dyn
    assert dyn.avatar() is not None and dyn.avatar()[0] == AVATAR
    assert {dyn.direction(a.value) for a in DIRS} == {(4, 0), (-4, 0), (0, -4), (0, 4)}
    assert used < 3000, "model agent did not finish both levels"
    assert agent.stats["skipped"] + agent.stats["explore_plans"] > 0
    text = "\n".join(dyn.describe())
    assert "colour 3 blocks the avatar" in text


def test_model_agent_is_not_worse_than_graph_agent_on_room():
    g = [run(GraphAgent(seed=s), Room(), 3000) for s in range(3)]
    m = [run(ModelAgent(seed=s), Room(), 3000) for s in range(3)]
    assert sum(m) <= sum(g)
