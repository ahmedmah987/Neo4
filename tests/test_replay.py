"""Three buttons must be clicked in order to complete a level; the next level has the
same buttons elsewhere. Exploration finds the order once; replay should reuse it."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.graph_agent import GraphAgent
from neo4.model_agent import ModelAgent

ORDER = [6, 7, 8]


class ButtonRoom:
    LAYOUTS = [
        {6: (10, 10), 7: (40, 12), 8: (24, 40)},
        {6: (44, 44), 7: (8, 30), 8: (30, 6)},
        {6: (20, 50), 7: (50, 20), 8: (6, 6)},
    ]

    def __init__(self, levels: int = 3):
        self.levels = levels
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self.progress = 0

    def _obs(self) -> FrameDataRaw:
        o = FrameDataRaw(state=self.state, levels_completed=self.done, win_levels=self.levels,
                         available_actions=[0, 6])
        if self.state in (GameState.NOT_FINISHED, GameState.WIN):
            g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
            g[60:64, :] = 3
            g[60:64, 0 : 20 * (self.progress + 1)] = 12  # a progress bar at the bottom
            layout = self.LAYOUTS[self.done % len(self.LAYOUTS)]
            for colour, (x, y) in layout.items():
                g[y : y + 4, x : x + 4] = colour
            o.frame = [g]
        return o

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self.progress = 0
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action is not GameAction.ACTION6:
            return self._obs()
        x, y = data["x"], data["y"]
        layout = self.LAYOUTS[self.done % len(self.LAYOUTS)]
        hit = next((c for c, (bx, by) in layout.items() if bx <= x < bx + 4 and by <= y < by + 4), None)
        if hit is None:
            return self._obs()
        if hit == ORDER[self.progress]:
            self.progress += 1
            if self.progress == len(ORDER):
                self.done += 1
                self.progress = 0
                if self.done == self.levels:
                    self.state = GameState.WIN
        else:
            self.progress = 0
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


def test_solution_replay_makes_later_levels_cheap():
    for cls in (GraphAgent, ModelAgent):
        agent = cls(seed=0)
        agent.max_replays = 2  # off by default: measured a loss on the public games
        steps = level_steps(agent, ButtonRoom(), 3000)
        assert len(steps) == 3, (cls.__name__, steps, agent.stats)
        assert agent.stats["replays"] >= 2 and agent.stats["replay_steps"] >= 6, agent.stats
        assert steps[1] - steps[0] <= 4 and steps[2] - steps[1] <= 4, (cls.__name__, steps)
        assert len(agent.solutions[0]) == 3 and all(t is not None for _, t, _ in agent.solutions[0])
