"""Adjacency goals: a level is completed by standing next to the colour-2 block and
pressing ACTION5. The block is never entered, so entry goals cannot express it; being
beside it, plus the key that worked before, can."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.model_agent import ModelAgent

WALL, BLOCK, AVATAR = 3, 2, 9
DIRS = {GameAction.ACTION1: (1, 0), GameAction.ACTION2: (-1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)}


class InteractRoom:
    blocks = [(9, 4), (5, 11), (12, 9)]

    def __init__(self):
        self.level, self.state = 0, GameState.NOT_PLAYED
        self.pos = (1, 1)

    def _obs(self):
        o = FrameDataRaw(state=self.state, levels_completed=self.level, win_levels=3,
                         available_actions=[0, 1, 2, 3, 4, 5])
        g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
        g[0:4, :] = g[60:64, :] = g[:, 0:4] = g[:, 60:64] = WALL
        bx, by = self.blocks[min(self.level, 2)]
        g[4 * by : 4 * by + 4, 4 * bx : 4 * bx + 4] = BLOCK
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
        bx, by = self.blocks[min(self.level, 2)]
        if action is GameAction.ACTION5:
            if abs(self.pos[0] - bx) + abs(self.pos[1] - by) == 1:
                self.level += 1
                if self.level == 3:
                    self.state = GameState.WIN
                else:
                    self.pos = (1, 1)
            return self._obs()
        dx, dy = DIRS.get(action, (0, 0))
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if nx in (0, 15) or ny in (0, 15) or (nx, ny) == (bx, by):
            return self._obs()
        self.pos = (nx, ny)
        return self._obs()


def run(agent, steps=4000):
    env = InteractRoom()
    obs = env.reset()
    solved = []
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.levels_completed > len(solved):
            solved.append(used)
        if obs.state == GameState.WIN:
            break
    return solved


def test_block_adjacency_and_the_key_are_learned_as_goal_conditions():
    agent = ModelAgent(seed=0, adjacency=True)
    solved = run(agent)
    assert len(solved) == 3, solved
    assert any(s[0] == BLOCK for s in agent.dyn.goal_adjacent_sigs()), agent.dyn.describe()
    assert GameAction.ACTION5.value in agent.dyn.goal_key_actions()
    first, later = solved[0], solved[-1] - solved[0]
    assert later < first, (first, later)


def test_adjacency_goals_speed_up_later_levels():
    with_adj = [run(ModelAgent(seed=s, adjacency=True)) for s in range(3)]
    without = [run(ModelAgent(seed=s, adjacency=False)) for s in range(3)]
    later = lambda runs: sum((r[-1] - r[0]) for r in runs if len(r) >= 2) + 4000 * sum(len(r) < 3 for r in runs)
    assert all(len(r) == 3 for r in with_adj), with_adj
    assert later(with_adj) < later(without), (with_adj, without)
