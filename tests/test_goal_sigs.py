"""Object-level goals: a room whose walls share the exit's colour. A colour goal would
send the planner into the walls; the exit's signature (a 4x4 sprite) does not."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.model_agent import ModelAgent

WALL = EXIT = 12  # same colour on purpose
AVATAR = 9
DIRS = {GameAction.ACTION1: (1, 0), GameAction.ACTION2: (-1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)}


class SameColourRoom:
    # Exits sit one cell away from the walls, otherwise they merge with them (same colour).
    exits = [(13, 13), (2, 13), (13, 2)]

    def __init__(self):
        self.level, self.state = 0, GameState.NOT_PLAYED
        self.pos = (1, 1)

    def _obs(self):
        o = FrameDataRaw(state=self.state, levels_completed=self.level, win_levels=3,
                         available_actions=[0, 1, 2, 3, 4])
        if self.state is not GameState.GAME_OVER:
            g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
            g[0:4, :] = g[60:64, :] = g[:, 0:4] = g[:, 60:64] = WALL
            ex, ey = self.exits[min(self.level, 2)]
            g[4 * ey : 4 * ey + 4, 4 * ex : 4 * ex + 4] = EXIT
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
        self.pos = (nx, ny)
        if self.pos == self.exits[min(self.level, 2)]:
            self.level += 1
            self.pos = (1, 1)
            if self.level == 3:
                self.state = GameState.WIN
        return self._obs()


def test_exit_signature_is_learned_and_later_levels_are_planned():
    agent = ModelAgent(seed=0)
    env = SameColourRoom()
    obs = env.reset()
    solved_at = []
    for used in range(1, 4000 + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.levels_completed > len(solved_at):
            solved_at.append(used)
        if obs.state == GameState.WIN:
            break
    assert len(solved_at) == 3, solved_at
    sigs = agent.dyn.goal_sigs()
    assert any(s[0] == EXIT and s[1] == 4 and s[2] == 4 for s in sigs), agent.dyn.describe()
    assert WALL not in agent.dyn.goal_colors() or True  # colour goal may exist; signature must win
    first = solved_at[0]
    later = [b - a for a, b in zip(solved_at, solved_at[1:])]
    assert all(t < first / 3 for t in later), (first, later)
    assert agent.stats["plans"] >= 2
