"""A room with an energy bar in the status area: every move costs one unit, a tile
refills it, and the exit is farther than one bar's worth of moves. Levels differ in
layout, so the second level is where the agent must plan through the refill tile."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.model_agent import ModelAgent

STEP = 2
ENERGY = 16


class EnergyRoom:
    # (avatar start, refill tile, exit) per level; the exit is 24 moves from the start,
    # 12 from the refill tile, which is 12 from the start
    LAYOUTS = [((4, 4), (16, 16), (28, 28)), ((28, 4), (16, 16), (4, 28))]

    def __init__(self, levels: int = 2):
        self.levels = levels
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self._place(0)

    def _place(self, level: int) -> None:
        self.pos, self.refill, self.exit = self.LAYOUTS[level % len(self.LAYOUTS)]
        self.energy = ENERGY

    def _obs(self) -> FrameDataRaw:
        o = FrameDataRaw(state=self.state, levels_completed=self.done, win_levels=self.levels,
                         available_actions=[0, 1, 2, 3, 4])
        if self.state in (GameState.NOT_FINISHED, GameState.WIN):
            g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
            g[0:2, :] = 3; g[32:34, :] = 3; g[:, 0:2] = 3; g[:, 32:34] = 3  # walls of a 34x34 room
            g[34:61, :] = 3; g[:, 34:] = 3
            g[61:63, :] = 5  # the status bar: empty part
            g[61:63, 0 : 2 * self.energy] = 11  # the filled part, two pixels per unit
            g[63, :] = 3
            rx, ry = self.refill
            g[ry : ry + 2, rx : rx + 2] = 6  # the refill tile
            ex, ey = self.exit
            g[ey : ey + 2, ex : ex + 2] = 12  # the exit
            x, y = self.pos
            g[y : y + 2, x : x + 2] = 9  # the avatar
            o.frame = [g]
        return o

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._place(self.done)
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs()
        d = {GameAction.ACTION1: (0, -STEP), GameAction.ACTION2: (0, STEP),
             GameAction.ACTION3: (-STEP, 0), GameAction.ACTION4: (STEP, 0)}.get(action)
        if d is None:
            return self._obs()
        if self.energy == 0:
            self.state = GameState.GAME_OVER
            return self._obs()
        self.energy -= 1
        nx, ny = self.pos[0] + d[0], self.pos[1] + d[1]
        if nx < 2 or ny < 2 or nx > 30 or ny > 30:
            return self._obs()  # a wall; the move still cost energy
        self.pos = (nx, ny)
        if self.pos == self.refill:
            self.energy = ENERGY
        if self.pos == self.exit:
            self.done += 1
            if self.done == self.levels:
                self.state = GameState.WIN
            else:
                self._place(self.done)
        return self._obs()


def level_steps(agent, env, steps):
    obs = env.reset()
    out = []
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(out) < obs.levels_completed:
            out.append(used)
        if obs.state == GameState.WIN:
            break
    return out


def test_model_agent_reads_the_energy_bar_and_refills_on_the_way():
    agent = ModelAgent(seed=0)
    steps = level_steps(agent, EnergyRoom(), 8000)
    dyn = agent.dyn
    res = dyn.resource()
    assert res is not None and res[0] == 11 and res[1] == 4, res
    assert any(t[0] == 6 for t in dyn.refill_tiles()), dyn.refill
    assert len(steps) == 2, (steps, agent.stats)
    assert agent.stats["refill_plans"] >= 1
    assert steps[1] - steps[0] < 120, (steps, agent.stats)
    assert "is a resource" in "\n".join(dyn.describe())
