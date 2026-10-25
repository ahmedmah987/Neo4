"""The library's hand-over: a solver keeps the turn while it wants it, and through the restart
that follows a go of its that ended (a death, or a step bar run out)."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent


class Stub:
    """A solver that wants every turn once it has seen ``start`` observations, pressing ``key``."""

    def __init__(self, name: str, key: GameAction, start: int) -> None:
        self.name, self.key, self.start = name, key, start
        self.disabled = False
        self.stats: dict = {}
        self.seen = self.turns = 0

    def observe(self, action, data, obs) -> None:
        self.seen += 1

    def act(self, obs):
        if self.seen < self.start:
            return None
        self.turns += 1
        return self.key, None


def frame(state: GameState) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=0, win_levels=1, available_actions=[1, 2])
    o.frame = [np.zeros((64, 64), dtype=np.int8)]
    return o


def test_the_solver_whose_go_ended_is_asked_first_after_the_restart():
    early = Stub("early", GameAction.ACTION1, start=3)  # first in the order; wants turns from the restart on
    late = Stub("late", GameAction.ACTION2, start=0)  # plays from the start
    agent = LibraryAgent(seed=0, families=[])
    agent.solvers = [early, late]
    assert agent.act(frame(GameState.NOT_FINISHED))[0] is GameAction.ACTION2
    assert agent.act(frame(GameState.NOT_FINISHED))[0] is GameAction.ACTION2
    # the go ends under the late solver, and the fallback begins the level again
    assert agent.act(frame(GameState.GAME_OVER))[0] is GameAction.RESET
    assert agent.controller is None
    # after the restart the late solver is asked first, though the early one wants the turn
    assert agent.act(frame(GameState.NOT_FINISHED))[0] is GameAction.ACTION2
    assert agent.controller is late and early.turns == 0
    # once the late solver gives the turn up, the order is the usual one again
    late.start = 10**9
    assert agent.act(frame(GameState.NOT_FINISHED))[0] is GameAction.ACTION1
    assert agent.controller is early
