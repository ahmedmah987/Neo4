"""Model-guided exploration: the graph agent plus induced rules.

Every executed transition is fed to ``Dynamics``. Before testing the
candidates of a node, the agent asks the model about each one:

* a candidate predicted to do nothing (walking into a known wall, clicking a
  signature that never changed anything) or to lose the game is counted as
  tested without being executed — in the first epoch only, so a wrong rule
  costs at most one re-test when the level graph is reopened;
* among the rest, candidates whose outcome the model cannot predict are
  tested before those it can, so exploration goes where the knowledge is thin.

Once a level has been completed, the model also knows what did it: the colour
the avatar entered, or the signature that was clicked. In later levels the
agent plans a path to that colour through cells it knows to be passable
(``planner``) and follows it while every step lands where the model said it
would; a surprise abandons the plan and hands control back to exploration.
Clicks on a signature that completed a level before are tested first.

Objects the model classes as *noise* — signatures that move, appear or vanish
whatever the agent does, such as drifting particles or an in-scene timer —
are erased from the frame before its state key is computed, so that their
motion does not multiply the number of states. When the noise set changes,
the current level's graph is rebuilt under the new identity.
"""

from __future__ import annotations

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp
from .dynamics import GAME_OVER, LEVEL_UP, OK, Dynamics
from .graph_agent import GraphAgent, LevelGraph, Node
from .objects import Obj, at, erase, extract, find
from .planner import plan_path

SKIP_VERDICTS = ("blocked", "danger", "noop_click")
MAX_PLANS_PER_LEVEL = 5
PLAN_COOLDOWN = 20  # exploration steps after a failed plan before planning again


class ModelAgent(GraphAgent):
    name = "model"

    def __init__(
        self,
        seed: int = 0,
        skip: bool = True,
        plan: bool = True,
        abstract: bool = False,
        curious: bool = True,
    ) -> None:
        super().__init__(seed)
        self.dyn = Dynamics()
        self.skip = skip
        self.planning = plan
        self.abstract = abstract
        self.curious = curious
        self.noise: frozenset = frozenset()
        self.rebuilt_this_level = False
        self.last_cand: tuple[GameAction, dict[str, int] | None] | None = None
        self.last_grid: np.ndarray | None = None
        self.last_objs: list[Obj] | None = None
        self.cur_grid: np.ndarray | None = None
        self._cur_objs: list[Obj] | None = None
        self._verdicts: dict[int, str] = {}
        self.plan: list[int] = []
        self.plan_expect: list[tuple[int, int]] = []
        self.plans_this_level = 0
        self.plan_cooldown = 0
        self.stats.update(
            {"skipped": 0, "ran_unknown": 0, "ran_predicted": 0,
             "plans": 0, "plan_steps": 0, "plan_failures": 0, "goal_clicks": 0,
             "noise_sigs": 0, "graph_rebuilds": 0}
        )

    def _key(self, grid: np.ndarray) -> str:
        if not self.abstract:
            return super()._key(grid)
        noise = self.dyn.noise()
        if noise != self.noise:
            # The state identity changed: the current level's graph is keyed the old
            # way. Rebuild it once per level; later growth of the noise set only
            # strands a few stale nodes, which is cheaper than starting over again.
            self.noise = noise
            self.stats["noise_sigs"] = len(noise)
            if self.graph.nodes and not self.rebuilt_this_level:
                self.graph = LevelGraph()
                self.first_key = None
                self.last_key = self.last_idx = None
                self.priority = 0
                self.rebuilt_this_level = True
                self.stats["graph_rebuilds"] += 1
        if not noise:
            return super()._key(grid)
        return fp.state_key(erase(grid, self.cur_objs(), noise), self.mask)

    def _start_level(self, grid: np.ndarray) -> None:
        super()._start_level(grid)
        self.plan = []
        self.plan_expect = []
        self.plans_this_level = 0
        self.plan_cooldown = 0
        self.rebuilt_this_level = False

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
        else:
            after = None
        self.cur_grid = grid
        # Within a level the mask is fixed, so this frame's objects are the ones just extracted.
        self._cur_objs = after
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
        if self.cur_grid is None or not self.dyn.observations:
            return super()._choose(node)
        objs = self.cur_objs()
        if self.planning:
            idx = self._follow_or_make_plan(node, objs)
            if idx is not None:
                return idx
        for i in node.untested(self.graph.max_group(), self.graph.epoch):
            c = node.cands[i]
            verdict = self.dyn.predict(c.action, c.data, objs, self.cur_grid).verdict
            if self.skip and self.graph.epoch == 0 and verdict in SKIP_VERDICTS:
                self.graph.mark_tested(node.key, i)
                self.stats["skipped"] += 1
            else:
                self._verdicts[i] = verdict
        node.interest = sum(1 for v in self._verdicts.values() if v == "unknown")
        return super()._choose(node)

    # -- planning -------------------------------------------------------------

    def _follow_or_make_plan(self, node: Node, objs: list[Obj]) -> int | None:
        avatar = self.dyn.avatar()
        me = find(objs, avatar) if avatar is not None else None
        if self.plan:
            expect = self.plan_expect[0]
            if me is None or (me.x1, me.y1) != expect:
                self.plan = []
                self.plan_expect = []
                self.plan_cooldown = PLAN_COOLDOWN
                self.stats["plan_failures"] += 1
            else:
                self.plan.pop(0)
                self.plan_expect.pop(0)
        if not self.plan:
            if self.plan_cooldown > 0:
                self.plan_cooldown -= 1
                return None
            if me is None or self.plans_this_level >= MAX_PLANS_PER_LEVEL:
                return None
            goals = self.dyn.goal_colors()
            dirs = self.dyn.directions()
            if not goals or not dirs:
                return None
            assert self.cur_grid is not None
            found = plan_path(
                self.cur_grid, me, dirs, self.dyn.passable, lambda cols: bool(cols & goals)
            )
            self.plans_this_level += 1
            if found is None:
                self.plan_cooldown = PLAN_COOLDOWN
                return None
            self.plan, self.plan_expect = found
            self.stats["plans"] += 1
        aid = self.plan[0]
        idx = next((i for i, c in enumerate(node.cands) if c.action.value == aid), None)
        if idx is None:
            self.plan = []
            self.plan_expect = []
            return None
        self.stats["plan_steps"] += 1
        return idx

    def _pick_local(self, node: Node, local: list[int]) -> int:
        goal_sigs = self.dyn.goal_click_sigs() if self.planning else set()
        if goal_sigs:
            objs = self.cur_objs()
            winners = [
                i for i in local
                if node.cands[i].data is not None
                and (t := at(objs, node.cands[i].data["x"], node.cands[i].data["y"])) is not None
                and t.sig in goal_sigs
            ]
            if winners:
                self.stats["goal_clicks"] += 1
                return self.rng.choice(winners)
        unknown = [i for i in local if self._verdicts.get(i, "unknown") == "unknown"]
        if unknown:
            self.stats["ran_unknown"] += 1
            return self.rng.choice(unknown)
        self.stats["ran_predicted"] += 1
        return self.rng.choice(local)


class ModelNoPlanAgent(ModelAgent):
    """Ablation: the induced rules prune and order candidates, but no planning."""

    name = "model-noplan"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed, skip=True, plan=False)


class ModelAbstractAgent(ModelAgent):
    """Ablation: the model agent with state abstraction over noise objects turned on."""

    name = "model-abs"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed, skip=True, plan=True, abstract=True)
