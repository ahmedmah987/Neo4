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
MAX_EXPLORE_PLANS_PER_LEVEL = 40
PLAN_COOLDOWN = 20  # exploration steps after a failed plan before planning again


class ModelAgent(GraphAgent):
    name = "model"

    def __init__(
        self,
        seed: int = 0,
        skip: bool = True,
        plan: bool = True,
        abstract: bool = False,
        curious: bool = False,
        explore: bool = False,
        adjacency: bool = False,
    ) -> None:
        super().__init__(seed)
        self.dyn = Dynamics()
        self.skip = skip
        self.planning = plan
        self.abstract = abstract
        self.curious = curious
        self.explore = explore
        self.adjacency = adjacency
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
        self.plan_kind = ""
        self.plans_this_level = 0
        self.explore_plans_this_level = 0
        self.plan_cooldown = 0
        self.stats.update(
            {"skipped": 0, "ran_unknown": 0, "ran_predicted": 0,
             "plans": 0, "plan_steps": 0, "plan_failures": 0, "goal_clicks": 0,
             "explore_plans": 0, "explore_steps": 0,
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
        self._drop_plan()
        self.plans_this_level = 0
        self.explore_plans_this_level = 0
        self.plan_cooldown = 0
        self.rebuilt_this_level = False

    def _drop_plan(self) -> None:
        self.plan = []
        self.plan_expect = []
        self.plan_kind = ""

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
            self._drop_plan()
        else:
            self.last_cand = (action, data)
            self.last_grid = grid
            self.last_objs = self.cur_objs()
        return action, data

    def _choose(self, node: Node) -> int | None:
        if self.cur_grid is None or not self.dyn.observations:
            return super()._choose(node)
        objs = self.cur_objs()
        avatar = self.dyn.avatar()
        me = find(objs, avatar) if avatar is not None else None
        if self.planning:
            idx = self._follow_plan(node, me)
            if idx is None and not self.plan:
                idx = self._make_plan(node, me, "goal")
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
        if self.explore and node.interest == 0 and not self.plan:
            # Nothing unknown to test here: go and touch something the avatar has not
            # touched yet, rather than grinding through moves whose outcome is known.
            idx = self._make_plan(node, me, "explore")
            if idx is not None:
                return idx
        return super()._choose(node)

    # -- planning -------------------------------------------------------------

    def _follow_plan(self, node: Node, me: Obj | None) -> int | None:
        """Next step of the current plan, after checking the last one landed as predicted."""
        if not self.plan:
            return None
        expect = self.plan_expect[0]
        if me is None or (me.x1, me.y1) != expect:
            self._drop_plan()
            self.plan_cooldown = PLAN_COOLDOWN
            self.stats["plan_failures"] += 1
            return None
        self.plan.pop(0)
        self.plan_expect.pop(0)
        if not self.plan:
            return None
        return self._plan_step(node)

    def _plan_step(self, node: Node) -> int | None:
        aid = self.plan[0]
        idx = next((i for i, c in enumerate(node.cands) if c.action.value == aid), None)
        if idx is None:
            self._drop_plan()
            return None
        if len(self.plan) == 1:
            # The final move touches the target; whether it goes through or not, the
            # plan is done, so do not hold the next step to a predicted position.
            self.plan_expect[0] = (-1, -1)
        self.stats["plan_steps" if self.plan_kind == "goal" else "explore_steps"] += 1
        return idx

    def _make_plan(self, node: Node, me: Obj | None, kind: str) -> int | None:
        if self.plan_cooldown > 0:
            self.plan_cooldown -= 1
            return None
        if me is None or self.cur_grid is None:
            return None
        dirs = self.dyn.directions()
        if not dirs:
            return None
        if kind == "goal":
            if self.plans_this_level >= MAX_PLANS_PER_LEVEL:
                return None
            # Object-level goals first (the exit sprite), then the halo around objects
            # that were beside the avatar when a level ended, colour goals last.
            goal_sigs = self.dyn.goal_sigs()
            target = np.zeros(self.cur_grid.shape, dtype=bool)
            for o in self.cur_objs():
                if o.sig in goal_sigs:
                    target[o.pixels[:, 0], o.pixels[:, 1]] = True
            if not target.any() and self.adjacency:
                near_sigs = self.dyn.goal_adjacent_sigs()
                near = [o for o in self.cur_objs() if o.sig in near_sigs]
                if near and not any(self._touching(me, o) for o in near):
                    target = self._halo(near, me)
            if target.any():
                is_goal = lambda cols, coords: any(target[y, x] for y, x in coords)
            else:
                goals = self.dyn.goal_colors()
                if not goals:
                    return None
                is_goal = lambda cols, coords: bool(cols & goals)
        else:
            if self.explore_plans_this_level >= MAX_EXPLORE_PLANS_PER_LEVEL:
                return None
            target = np.zeros(self.cur_grid.shape, dtype=bool)
            for o in self.cur_objs():
                if not o.scenery and o.sig != me.sig and o.sig not in self.dyn.touched:
                    target[o.pixels[:, 0], o.pixels[:, 1]] = True
            if not target.any():
                return None
            is_goal = lambda cols, coords: any(target[y, x] for y, x in coords)
        found = plan_path(self.cur_grid, me, dirs, self.dyn.passable, is_goal)
        if found is None:
            # No route through cells known to be passable yet; try again after a
            # cooldown, and do not spend the per-level allowance on the search.
            self.plan_cooldown = PLAN_COOLDOWN
            return None
        if kind == "goal":
            self.plans_this_level += 1
        else:
            self.explore_plans_this_level += 1
        self.plan, self.plan_expect = found
        self.plan_kind = kind
        self.stats["plans" if kind == "goal" else "explore_plans"] += 1
        return self._plan_step(node)

    @staticmethod
    def _touching(me: Obj, o: Obj) -> bool:
        return o.x1 <= me.x2 + 1 and o.x2 >= me.x1 - 1 and o.y1 <= me.y2 + 1 and o.y2 >= me.y1 - 1

    def _halo(self, objs: list[Obj], me: Obj) -> np.ndarray:
        """Cells from which the avatar would touch one of ``objs``: a band as wide as
        the avatar's own step around each object, excluding the objects themselves."""
        assert self.cur_grid is not None
        h, w = self.cur_grid.shape
        halo = np.zeros((h, w), dtype=bool)
        band = max(me.w, me.h)
        for o in objs:
            y1, y2 = max(o.y1 - band, 0), min(o.y2 + band, h - 1)
            x1, x2 = max(o.x1 - band, 0), min(o.x2 + band, w - 1)
            halo[y1 : y2 + 1, x1 : x2 + 1] = True
        for o in objs:
            halo[o.pixels[:, 0], o.pixels[:, 1]] = False
        return halo

    def _push_candidates(self, node: Node, local: list[int], me: Obj, objs: list[Obj]) -> list[int]:
        """Beside a goal-adjacent object: the key that completed levels before, or a
        move towards the object (a push)."""
        near_sigs = self.dyn.goal_adjacent_sigs()
        if not near_sigs or self.cur_grid is None:
            return []
        near = [o for o in objs if o.sig in near_sigs and self._touching(me, o)]
        if not near:
            return []
        keys = self.dyn.goal_key_actions()
        out = [i for i in local if node.cands[i].data is None and node.cands[i].action.value in keys]
        if out:
            return out
        for i in local:
            c = node.cands[i]
            if c.data is not None:
                continue
            d = self.dyn.direction(c.action.value)
            if d is None:
                continue
            for o in near:
                dx = (o.x1 + o.x2) / 2 - (me.x1 + me.x2) / 2
                dy = (o.y1 + o.y2) / 2 - (me.y1 + me.y2) / 2
                if d[0] * dx + d[1] * dy > 0 and abs(dx) * abs(d[0]) + abs(dy) * abs(d[1]) >= max(abs(dx), abs(dy)):
                    out.append(i)
                    break
        return out

    def _goal_candidates(self, node: Node, local: list[int]) -> list[int]:
        """Untested candidates that reproduce what completed a level before."""
        if not self.planning or not self.dyn.level_ups:
            return []
        objs = self.cur_objs()
        # Click goals by signature only: a goal by *colour* made the agent spend 500
        # steps per run clicking every colour-15 pixel of r11l, where position decides.
        goal_click = self.dyn.goal_click_sigs()
        goal_sigs = self.dyn.goal_sigs()
        goal_colors = self.dyn.goal_colors()
        avatar = self.dyn.avatar()
        me = find(objs, avatar) if avatar is not None else None
        out = []
        for i in local:
            c = node.cands[i]
            if c.data is not None:
                t = at(objs, c.data["x"], c.data["y"])
                if t is not None and t.esig in goal_click:
                    out.append(i)
            elif me is not None and (goal_sigs or goal_colors) and self.cur_grid is not None:
                d = self.dyn.direction(c.action.value)
                if d is None:
                    continue
                if goal_sigs and self.dyn.ahead_sigs(me, d, self.cur_grid, objs) & goal_sigs:
                    out.append(i)
                elif not goal_sigs and self.dyn._ahead(me, d, self.cur_grid) & goal_colors:
                    out.append(i)
        if not out and me is not None and self.adjacency:
            out = self._push_candidates(node, local, me, objs)
        return out

    def _pick_local(self, node: Node, local: list[int]) -> int:
        winners = self._goal_candidates(node, local)
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


class ModelCuriousAgent(ModelAgent):
    """Ablation: the default model agent with the curiosity-directed frontier."""

    name = "model-cur"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed, curious=True)


class ModelExploreAgent(ModelAgent):
    """Ablation: the default model agent plus object-targeted exploration."""

    name = "model-explore"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed, explore=True)


class ModelAdjacencyAgent(ModelAgent):
    """Ablation: the default model agent plus adjacency goal conditions."""

    name = "model-adj"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed, adjacency=True)
