"""A room whose exit admits the avatar only in one indicator state; a switch tile
cycles the indicator shown far away in a corner. Levels differ in layout, so the
second level is where the agent must learn to lock the goal and go for the switch."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.model_agent import ModelAgent
from neo4.graph_agent import GraphAgent

STEP = 2
STATES = 3


class SwitchRoom:
    # (avatar start, switch tile, exit) per level, all 2x2 on an even lattice
    LAYOUTS = [((8, 8), (12, 16), (24, 26)), ((20, 20), (14, 12), (8, 28))]

    def __init__(self, levels: int = 2):
        self.levels = levels
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self._place(0)

    def _place(self, level: int) -> None:
        self.pos, self.switch, self.exit = self.LAYOUTS[level % len(self.LAYOUTS)]
        self.indicator = 0
        self.on_switch = False

    def _obs(self) -> FrameDataRaw:
        o = FrameDataRaw(state=self.state, levels_completed=self.done, win_levels=self.levels,
                         available_actions=[0, 1, 2, 3, 4])
        if self.state in (GameState.NOT_FINISHED, GameState.WIN):
            g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
            g[0:2, :] = 3; g[34:36, :] = 3; g[:, 0:2] = 3; g[:, 34:36] = 3  # walls of a 34x34 room
            g[36:, :] = 3; g[:, 36:] = 3
            sx, sy = self.switch
            g[sy : sy + 2, sx : sx + 2] = 6  # the switch tile
            ex, ey = self.exit
            g[ey : ey + 2, ex : ex + 2] = 12  # the exit
            g[50:52, 40 + 4 * self.indicator : 42 + 4 * self.indicator] = 7  # the indicator, far away
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
        nx, ny = self.pos[0] + d[0], self.pos[1] + d[1]
        if nx < 2 or ny < 2 or nx > 32 or ny > 32:
            return self._obs()  # a wall
        if (nx, ny) == self.exit:
            if self.indicator != 1:
                return self._obs()  # locked: the exit blocks like a wall
            self.done += 1
            if self.done == self.levels:
                self.state = GameState.WIN
            else:
                self._place(self.done)
            return self._obs()
        self.pos = (nx, ny)
        if self.pos == self.switch:
            if not self.on_switch:
                self.indicator = (self.indicator + 1) % STATES
            self.on_switch = True
        else:
            self.on_switch = False
        return self._obs()


def run(agent, env, steps):
    obs = env.reset()
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.state == GameState.WIN:
            return used, obs
    return steps, obs


def level_steps(agent, env, steps):
    """Step at which each level was completed."""
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


def test_model_agent_learns_the_switch_and_unlocks_the_exit():
    agent = ModelAgent(seed=0)
    used, obs = run(agent, SwitchRoom(), 6000)
    assert obs.levels_completed == 2, (used, agent.stats)
    dyn = agent.dyn
    switch = next(t for t in dyn.switches() if t[0] == 6)
    assert switch is not None and any(sig[0] == 7 for sig in dyn.indicator_sigs())
    assert agent.stats["switch_plans"] >= 1 and agent.stats["locked_goals"] >= 1
    assert "is a switch" in "\n".join(dyn.describe())


def test_second_level_is_much_cheaper_with_the_switch_rule():
    """Both agents explore the first level; on the second the model agent plans to
    the exit, finds it locked, flips the switch and plans again."""
    graph = level_steps(GraphAgent(seed=0), SwitchRoom(), 8000)
    model = level_steps(ModelAgent(seed=0), SwitchRoom(), 8000)
    assert len(model) == 2
    model_second = model[1] - model[0]
    assert model_second < 200
    if len(graph) == 2:
        assert model_second * 3 < graph[1] - graph[0]
