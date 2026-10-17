"""A block is pushed towards a target by clicking a button (another button pushes it
back); the level ends when the block touches the target. The next level has a longer
way to go: difference reduction should make it a straight descent."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.model_agent import ModelAgent

STEP = 2


class SlideRoom:
    LAYOUTS = [(10, 24, 30), (6, 40, 44)]  # block x, target x, row y

    def __init__(self, levels: int = 2):
        self.levels = levels
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self._place(0)

    def _place(self, level: int) -> None:
        self.block, self.target, self.row = self.LAYOUTS[level % len(self.LAYOUTS)]
        # the decoy kinds change place every level so that no fixed relation to them holds
        self.decoy = (50 if level % 2 == 0 else 20, 50 if level % 2 == 0 else 8)

    def _obs(self) -> FrameDataRaw:
        o = FrameDataRaw(state=self.state, levels_completed=self.done, win_levels=self.levels,
                         available_actions=[0, 6])
        if self.state in (GameState.NOT_FINISHED, GameState.WIN):
            g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
            g[0:2, :] = 3; g[62:64, :] = 3; g[:, 0:2] = 3; g[:, 62:64] = 3
            g[2:6, 2:6] = 6  # the "right" button
            g[2:6, 58:62] = 7  # the "left" button
            g[self.row : self.row + 4, self.block : self.block + 4] = 9  # the block
            g[self.row : self.row + 4, self.target : self.target + 4] = 12  # the target
            dx, dy = self.decoy
            g[dy : dy + 3, dx : dx + 3] = 11  # a decoy that does nothing
            o.frame = [g]
        return o

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._place(self.done)
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action is not GameAction.ACTION6:
            return self._obs()
        x, y = data["x"], data["y"]
        if 2 <= x < 6 and 2 <= y < 6:
            self.block = min(self.block + STEP, self.target - 4)
        elif 58 <= x < 62 and 2 <= y < 6:
            self.block = max(self.block - STEP, 4)
        else:
            return self._obs()
        if self.block + 4 == self.target:
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


def test_difference_reduction_descends_to_the_target():
    agent = ModelAgent(seed=0)
    steps = level_steps(agent, SlideRoom(), 2000)
    assert len(steps) == 2, (steps, agent.stats)
    assert agent.relations and agent.relations[0][0][0] == 9 and agent.relations[0][1][0] == 12
    # The second level needs 17 pushes; a plain explorer tests every button at every
    # state and takes far longer than that.
    assert steps[1] - steps[0] <= 60, (steps, agent.stats)


def test_relation_kinds_survive_a_recolouring_between_levels():
    """The block and the target change colour on the second level; the relation
    learned on the first must still apply by size and shape."""

    class RecolouredSlideRoom(SlideRoom):
        def _obs(self):
            o = super()._obs()
            if o.frame is not None:
                g = o.frame[-1]
                # distinctive shapes: a ring for the block, a plus for the target
                g[self.row + 1 : self.row + 3, self.block + 1 : self.block + 3] = 0
                for dx, dy in ((0, 0), (3, 0), (0, 3), (3, 3)):
                    g[self.row + dy, self.target + dx] = 0
                if self.done % 2 == 1:
                    g[g == 9] = 10
                    g[g == 12] = 13
            return o

    agent = ModelAgent(seed=0)
    steps = level_steps(agent, RecolouredSlideRoom(), 2000)
    assert len(steps) == 2, (steps, agent.stats)
    assert agent.stats["gain_clicks"] >= 10, agent.stats
    assert steps[1] - steps[0] <= 60, (steps, agent.stats)
