"""Model-guided exploration: the graph agent plus induced rules.

Every executed transition is fed to ``Dynamics``. Before testing the
candidates of a node, the agent asks the model about each one:

* a candidate predicted to do nothing (walking into a known wall, clicking a
  signature that never changed anything) or to lose the game is counted as
  tested without being executed — in the first epoch only, so a wrong rule
  costs at most one re-test when the level graph is reopened;
* among the rest, candidates whose outcome the model cannot predict are
  tested before those it can, so exploration goes where the knowledge is thin.
"""

from __future__ import annotations

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from .dynamics import GAME_OVER, LEVEL_UP, OK, Dynamics
from .graph_agent import GraphAgent, Node
from .objects import Obj, extract

SKIP_VERDICTS = ("blocked", "danger", "noop_click")


class ModelAgent(GraphAgent):
    name = "model"

    def __init__(self, seed: int = 0, skip: bool = True) -> None:
        super().__init__(seed)
        self.dyn = Dynamics()
        self.skip = skip
        self.last_cand: tuple[GameAction, dict[str, int] | None] | None = None
        self.last_grid: np.ndarray | None = None
        self.last_objs: list[Obj] | None = None
        self.cur_grid: np.ndarray | None = None
        self._cur_objs: list[Obj] | None = None
        self._verdicts: dict[int, str] = {}
        self.stats.update({"skipped": 0, "ran_unknown": 0, "ran_predicted": 0})

    def cur_objs(self) -> list[Obj]:
        if self._cur_objs is None:
            assert self.cur_grid is not None
            self._cur_objs = extract(self.cur_grid, self.mask)
        return self._cur_objs

    def act(self, obs: FrameDataRaw) -> tuple[GameAction, dict[str, int] | None]:
        grid = None if obs.is_empty() else np.asarray(obs.frame[-1])
        if self.last_cand is not None and self.last_grid is not None and self.last_objs is not None:
            if obs.state == GameState.GAME_OVER:
                outcome = GAME_OVER
            elif obs.levels_completed > self.levels_completed:
                outcome = LEVEL_UP
            else:
                outcome = OK
            after = extract(grid, self.mask) if grid is not None and outcome == OK else None
            action, data = self.last_cand
            self.dyn.observe(action, data, self.last_objs, self.last_grid, after, outcome)
            self.last_cand = None
        self.cur_grid = grid
        self._cur_objs = None
        self._verdicts = {}

        action, data = super().act(obs)

        if action is GameAction.RESET or grid is None:
            self.last_cand = None
        else:
            self.last_cand = (action, data)
            self.last_grid = grid
            self.last_objs = self.cur_objs()
        return action, data

    def _choose(self, node: Node) -> int | None:
        if self.cur_grid is not None and self.dyn.observations:
            objs = self.cur_objs()
            for i in node.untested(self.graph.max_group(), self.graph.epoch):
                c = node.cands[i]
                verdict = self.dyn.predict(c.action, c.data, objs, self.cur_grid).verdict
                if self.skip and self.graph.epoch == 0 and verdict in SKIP_VERDICTS:
                    self.graph.mark_tested(node.key, i)
                    self.stats["skipped"] += 1
                else:
                    self._verdicts[i] = verdict
        return super()._choose(node)

    def _pick_local(self, node: Node, local: list[int]) -> int:
        unknown = [i for i in local if self._verdicts.get(i, "unknown") == "unknown"]
        if unknown:
            self.stats["ran_unknown"] += 1
            return self.rng.choice(unknown)
        self.stats["ran_predicted"] += 1
        return self.rng.choice(local)
