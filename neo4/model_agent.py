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

import os

from collections import Counter

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp
from .advisor import Advisor, build_prompt, describe_objects, listed_objects
from . import hypotheses as hy
from .effects import EffectModel, imagine
from .dynamics import GAME_OVER, LEVEL_UP, OK, Dynamics
from .graph_agent import GraphAgent, LevelGraph, Node
from .objects import Obj, at, erase, extract, find
from .planner import plan_path

SKIP_VERDICTS = ("blocked", "danger", "noop_click", "noop_key")
MAX_PLANS_PER_LEVEL = 5  # goal plans per level and avatar state
MAX_EXPLORE_PLANS_PER_LEVEL = 40
MAX_SWITCH_PLANS_PER_LEVEL = 30
PLAN_COOLDOWN = 20  # exploration steps after a failed plan before planning again
HORIZON = 40  # moves ahead over which a patrol's motion is extrapolated for planning
# The goal advisor (Phase 58) is asked once a level has taken this many actions without ending:
# not at its start, since advice followed there cost the cheap levels the explorer finds alone
# (NEO4_ADVICE_AT overrides, e.g. "0,40,250"); a game's questions are capped.
ADVICE_AT = tuple(int(a) for a in os.environ.get("NEO4_ADVICE_AT", "40,250").split(","))
MAX_ADVICE_PER_LEVEL = 5  # with the re-asks that follow advice carried out to its end
MAX_ADVICE_PER_GAME = 40
MAX_ADVICE_PLANS_PER_LEVEL = 12


class ModelAgent(GraphAgent):
    name = "model"
    # Clicks nearest to a moving kind first: measured neutral to harmful on the public
    # click games (r11l, lp85), so off unless a variant turns it on.
    near_clicks = False
    goal_counts = True  # count goals beside the gap relations (Phase 56)
    skip_undo = True  # the undo key, once seen to restore the frame before the last action, is not explored
    track_hypotheses = True  # goal hypotheses and the memory of what moves them (Phase 59)
    steer_confirmed = True  # steer by hypotheses of the kinds that ended earlier levels (Phase 59)
    steer_first = False  # before any level ends, steer by hypotheses some action moved closer
    steer_first_after = 300  # ...and only once the level has taken this many actions (when stuck)
    carry_causes = True  # what moved a goal of a type in earlier levels orders the first tries in the next
    narrow_confirmed = True  # steer by the confirmed kinds that ended the most levels
    imagine_clicks = False  # plan clicks in imagination towards a confirmed goal (Phase 60; a variant)
    experiment = False  # with no confirmed goal, test one hypothesis at a time (Phase 61; a variant)
    experiment_after = 30  # actions into a level before the first test
    experiment_patience = 40  # actions without a new best before a test is given up

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
        self.fatal: set[tuple[int, int, int]] = set()  # (x, y, action) moves that lost this level
        # Goal preconditions: indicator states in which the goal was reached and the
        # level did not end, the (state, switch) pairs tried, and goal plans per state.
        self.locked: set[frozenset] = set()
        self.tried_switch: set[tuple[frozenset, tuple]] = set()
        self.goal_plans_by_state: dict[frozenset, int] = {}
        self.switch_plans_this_level = 0
        self.switch_from: tuple[frozenset, tuple] | None = None
        # Difference reduction: the object kinds that came together when a level was
        # completed (a moving kind M and a static kind T); later levels are explored
        # towards frames where M is nearest to T.
        self.level_start_objs: list[Obj] | None = None
        self.relations: list[tuple[tuple, tuple]] = []
        # Which click-target kinds have closed the gap of the goal relation: the sum of
        # the gap changes their clicks produced, and how many were tried.
        self.kind_gain: dict[tuple, float] = {}
        self.kind_tries: dict[tuple, int] = {}
        self.h_first: dict[tuple, float] = {}
        self.h_kind: dict[tuple, tuple | None] = {}
        # Undo (the seventh action in ARC-AGI-3): how often it was tried after another action, and
        # how often the frame it gave back was the one from before that action. Only this key is
        # checked: a move left after a move right also gives the old frame back.
        self.undo_tries: dict[int, int] = {}
        self.undo_hits: dict[int, int] = {}
        self.keys_back: list[str | None] = [None, None]  # the frames' keys before the last two actions
        self.last_h: float | None = None
        self.stats.update(
            {"skipped": 0, "ran_unknown": 0, "ran_predicted": 0,
             "plans": 0, "plan_steps": 0, "plan_failures": 0, "goal_clicks": 0,
             "explore_plans": 0, "explore_steps": 0,
             "noise_sigs": 0, "graph_rebuilds": 0, "budget_deaths": 0, "fatal_moves": 0,
             "switch_plans": 0, "locked_goals": 0, "budget_resets": 0, "refill_plans": 0,
             "descents": 0, "near_clicks": 0, "gain_clicks": 0,
             "advice_queries": 0, "advice_answers": 0, "advice_plans": 0, "advice_steps": 0,
             "advice_clicks": 0}
        )
        # The goal advisor (Phase 58): None or an Advisor; what it advised in this level.
        self.advisor: Advisor | None = None
        self.available: list[int] = []
        self.hyps = None
        self.hyp_values = None
        self.click_plan = []
        self.click_expect = None
        self.imagined_this_level = 0
        self.imagined_known = -1
        self.level_actions = 0
        self.advice_queries = 0
        self.advice_next = 0  # the next scheduled question (ADVICE_AT)
        self.advice_asked_at = 0
        self.advice_targets: list[Obj] = []
        self.advice_done: set[int] = set()
        self.advice_current: int | None = None
        self.advice_plans = 0
        self.advice_tried: list[str] = []
        self.advice_log: list[str] = []
        # Goal hypotheses read from the screen and what each action class did to their measures
        # (Phase 59); hyp_reports keeps each completed level's hypotheses for diagnosis.
        self.hyps: list[hy.Hypothesis] | None = None
        self.hyp_values: dict[tuple, int | None] | None = None
        self.causes = hy.CauseMemory()
        self.hyp_reports: list[list[tuple]] = []
        self.confirmed: list[tuple] = []  # kinds of hypotheses that ended a level, most recent first
        self.confirm_count: Counter = Counter()  # how many levels each kind ended
        self.effects = EffectModel()  # what a click on each shape does (Phase 60)
        self.click_plan: list[Obj] = []  # imagined clicks still to make, in order
        self.click_expect: np.ndarray | None = None  # the frame the last planned click should give
        self.imagined_this_level = 0
        self.imagined_known = -1  # effects known at the last search: search again when it grows
        # Goal experiments (Phase 61): the hypothesis under test, its best value, the actions since
        # the last new best, the hypotheses of this level already tested, and each node's readings.
        self.trial: hy.Hypothesis | None = None
        self.trial_best: int | None = None
        self.trial_quiet = 0
        self.tried: set[tuple] = set()
        self.node_values: dict[str, dict] = {}
        self.node_rel: dict[str, float | None] = {}

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
        self.fatal = set()
        self.locked = set()
        self.tried_switch = set()
        self.goal_plans_by_state = {}
        self.switch_plans_this_level = 0
        self.switch_from = None
        self._cur_objs = None  # the mask changed with the level
        self.keys_back = [None, None]
        self.level_start_objs = list(self.cur_objs()) if self.cur_grid is not None else None
        self.best_first = bool(self.relations)
        self.h_first: dict[tuple, float] = {}  # each goal measure's value when first met this level
        self.h_kind: dict[tuple, tuple | None] = {}  # the kind a count goal stands for this level
        self.level_actions = 0
        self.advice_queries = 0
        self.advice_next = 0
        self.advice_asked_at = 0
        self.advice_targets = []
        self.advice_done = set()
        self.advice_current = None
        self.advice_plans = 0
        self.advice_tried = []
        self.click_plan = []
        self.click_expect = None
        self.imagined_this_level = 0
        self.imagined_known = -1
        self.trial = None
        self.trial_best = None
        self.trial_quiet = 0
        self.tried = set()
        self.node_values = {}
        self.node_rel = {}

    # -- difference reduction ---------------------------------------------------

    @staticmethod
    def _gap(a: Obj, b: Obj) -> int:
        dx = max(0, max(a.x1, b.x1) - min(a.x2, b.x2))
        dy = max(0, max(a.y1, b.y1) - min(a.y2, b.y2))
        return dx + dy

    @staticmethod
    def _summary(objs: list[Obj]) -> list[tuple]:
        return [(o.sig, o.x1, o.y1, o.x2, o.y2) for o in objs]

    @classmethod
    def _gap_in(cls, summary: list[tuple], m: tuple, t: tuple) -> int | None:
        ms = [s for s in summary if s[0] == m]
        ts = [s for s in summary if s[0] == t]
        if not ms or not ts:
            return None
        return min(
            max(0, max(a[1], b[1]) - min(a[3], b[3])) + max(0, max(a[2], b[2]) - min(a[4], b[4]))
            for a in ms for b in ts
        )

    def _learn_relations(self, start: list[Obj] | None, end: list[Obj] | None) -> None:
        """What the completed level asked, as goal hypotheses for the next levels.

        Gap relations: the kinds whose instances moved (or appeared) and ended nearest
        to a kind that stood still, ranked by how steadily their gap closed along the
        solution path (the shortest known route through the level graph), then by how
        much it closed; the two best pairs are kept. Counts (``goal_counts``): a kind or
        a colour whose number of objects moved steadily one way along the path, ending
        where it ended (all of a kind cleared, pegs taken, pieces added). Hypotheses an
        earlier level supported too are ranked first, so the agent's idea of the goal
        narrows from level to level."""
        if not start or not end:
            return
        s0: dict[tuple, list[Obj]] = {}
        s1: dict[tuple, list[Obj]] = {}
        for o in start:
            s0.setdefault(o.sig, []).append(o)
        for o in end:
            s1.setdefault(o.sig, []).append(o)
        places = lambda objs: sorted((o.x1, o.y1) for o in objs)
        moved = {k for k in s1 if k not in s0 or places(s0[k]) != places(s1[k])}
        static = {k for k in s0 if k in s1 and places(s0[k]) == places(s1[k])}
        edges = self._solution_edges() or []
        path = [self.graph.nodes[k].summary for k, _ in edges if self.graph.nodes[k].summary is not None]
        path.append(self._summary(end))
        scored = []
        for m in moved:
            if m not in s0:
                continue
            for t in static:
                g0 = min(self._gap(a, b) for a in s0[m] for b in s0[t])
                g1 = min(self._gap(a, b) for a in s1[m] for b in s1[t])
                if g1 >= g0:
                    continue
                gaps = [g for g in (self._gap_in(sm, m, t) for sm in path) if g is not None]
                down = sum(1 for a, b in zip(gaps, gaps[1:]) if b < a)
                up = sum(1 for a, b in zip(gaps, gaps[1:]) if b > a)
                steadiness = (down - up) / max(1, len(gaps) - 1)
                scored.append((steadiness, g0 - g1, g1, m, t))
        scored.sort(key=lambda r: (-r[0], -r[1], r[2]))
        hyps = [("gap", m, t) for _, _, _, m, t in scored[:2]]
        if self.goal_counts:
            counts = []
            series: dict[tuple, list[int]] = {}
            for i, summ in enumerate(path):
                seen: dict[tuple, int] = {}
                for sg in summ:
                    seen[("count", sg[0])] = seen.get(("count", sg[0]), 0) + 1
                    seen[("ncolour", sg[0][0])] = seen.get(("ncolour", sg[0][0]), 0) + 1
                for key in set(seen) | set(series):
                    series.setdefault(key, [0] * i).append(seen.get(key, 0))
            for key, vals in series.items():
                v0, v1 = vals[0], vals[-1]
                if v0 == v1:
                    continue
                d = 1 if v1 > v0 else -1
                toward = sum(1 for a, b in zip(vals, vals[1:]) if (b - a) * d > 0)
                away = sum(1 for a, b in zip(vals, vals[1:]) if (b - a) * d < 0)
                if away == 0:  # it only ever moved towards where it ended
                    counts.append((toward, abs(v1 - v0), key + (d, v1)))
            counts.sort(key=lambda r: (-r[0], -r[1]))
            hyps += [h for _, _, h in counts[:2]]
        if hyps:
            before = {self._goal_id(h) for h in self.relations}
            hyps.sort(key=lambda h: self._goal_id(h) not in before)  # stable: earlier support first
            self.relations = hyps
            self.best_first = True

    @staticmethod
    def _goal_id(h: tuple) -> tuple:
        """A goal hypothesis up to recolouring: kinds by size and shape."""
        if h[0] == "gap":
            return ("gap", h[1][1:], h[2][1:])
        if h[0] == "count":
            return ("count", h[1][1:], h[2])
        return h[:3]

    @staticmethod
    def _instances(objs: list[Obj], sig: tuple) -> list[Obj]:
        """Objects of a kind; when none has exactly this signature, the objects of the
        one kind with the same size and shape in another colour (levels recolour
        their pieces: vc33's markers are colour 11 on one level and 14 on the next)."""
        exact = [o for o in objs if o.sig == sig]
        if exact:
            return exact
        shape = sig[1:]
        kinds = {o.sig for o in objs if o.sig[1:] == shape and not o.scenery}
        if len(kinds) == 1:
            k = next(iter(kinds))
            return [o for o in objs if o.sig == k]
        return []

    def _kind_of(self, o: Obj, sigs) -> tuple | None:
        """The learned kind an object counts as: its own signature, or one with the
        same size and shape in another colour."""
        for sig in sigs:
            if o.sig == sig:
                return sig
        for sig in sigs:
            if o.sig[1:] == sig[1:]:
                return sig
        return None

    def _measure(self, h: tuple, objs: list[Obj]) -> float | None:
        """How far a frame is from a goal hypothesis (0: met), or None if it cannot say."""
        if h[0] == "gap":
            ms = self._instances(objs, h[1])
            ts = self._instances(objs, h[2])
            return float(min(self._gap(a, b) for a in ms for b in ts)) if ms and ts else None
        if h[0] == "count":
            if h not in self.h_kind:
                # the kind it stands for in this level: itself, or the one kind of its size
                # and shape in another colour, as the level's first frame shows them
                found = self._instances(self.level_start_objs or objs, h[1])
                self.h_kind[h] = found[0].sig if found else None
            if self.h_kind[h] is None:
                return None
            n = sum(1 for o in objs if o.sig == self.h_kind[h])
        else:
            n = sum(1 for o in objs if o.color == h[1])
        first = self.h_first.setdefault(h, float(n))
        if h[2] < 0:
            return float(n)  # fewer is better: all of them gone, at best
        return float(max(0, max(h[3], first + 1) - n))  # more is better, up to what the last level reached

    def _heuristic(self, objs: list[Obj]) -> float | None:
        """Over the goal hypotheses, how far a frame is from each, as a share of how far
        the level started from it; plus the screen hypotheses steered by (Phase 59)."""
        extra = self._hyp_heuristic() if self.track_hypotheses else None
        return self._combine(self._rel_heuristic(objs), extra)

    @staticmethod
    def _combine(a: float | None, b: float | None) -> float | None:
        return None if a is None and b is None else (a or 0.0) + (b or 0.0)

    def _rel_heuristic(self, objs: list[Obj]) -> float | None:
        """The goal relations learned at level-ups (Phase 22): the sum of their shares."""
        if not self.relations:
            return None
        total = 0.0
        found = False
        for h in self.relations:
            v = self._measure(h, objs)
            if v is None:
                continue
            key = ("scale",) + h
            scale = self.h_first.setdefault(key, max(1.0, v))
            total += v / scale
            found = True
        return total if found else None

    def _near_moving(self, node: Node, local: list[int], objs: list[Obj]) -> list[int]:
        """Click candidates nearest to an instance of a moving kind of the relation."""
        ms = [o for h in self.relations if h[0] == "gap" for o in objs if o.sig == h[1]]
        clicks = [i for i in local if node.cands[i].data is not None]
        if not ms or not clicks:
            return []

        def dist(i: int) -> int:
            x, y = node.cands[i].data["x"], node.cands[i].data["y"]
            return min(max(0, o.x1 - x, x - o.x2) + max(0, o.y1 - y, y - o.y2) for o in ms)

        best = min(dist(i) for i in clicks)
        return [i for i in clicks if dist(i) <= best]

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
        after: list[Obj] | None = None
        if self.last_cand is not None and self.last_grid is not None and self.last_objs is not None:
            if obs.state == GameState.GAME_OVER:
                outcome = GAME_OVER
            elif obs.levels_completed > self.levels_completed:
                outcome = LEVEL_UP
            else:
                outcome = OK
            # The frame of a lost game is kept: it usually still shows the scene,
            # with whatever killed the avatar standing where the avatar went.
            after = extract(grid, self.mask) if grid is not None and outcome != LEVEL_UP else None
            action, data = self.last_cand
            avatar = self.dyn.avatar()
            me = self.dyn.body(find(self.last_objs, avatar), self.last_objs) if avatar is not None else None
            self.dyn.observe(
                action, data, self.last_objs, self.last_grid, after, outcome,
                after_grid=grid, mask=self.mask,
            )
            if (outcome == OK and data is not None and grid is not None and self.last_grid is not None
                    and grid.shape == self.last_grid.shape):
                target = at(self.last_objs, data["x"], data["y"])
                if target is not None:
                    self.effects.observe(target, self.last_grid, grid, self.mask)
            if self.track_hypotheses:
                self._track_hypotheses(action, data, outcome, after, grid)
            if outcome == LEVEL_UP:
                self._learn_relations(self.level_start_objs, self.last_objs)
            elif outcome == OK and self.relations and data is not None and after is not None:
                # Credit the clicked kind with the change in the goal gap it produced.
                h_now = self._heuristic(after)
                target = at(self.last_objs, data["x"], data["y"])
                if h_now is not None and self.last_h is not None and target is not None:
                    kind = self._kind_of(target, self.kind_tries) or target.sig
                    self.kind_gain[kind] = self.kind_gain.get(kind, 0.0) + (h_now - self.last_h)
                    self.kind_tries[kind] = self.kind_tries.get(kind, 0) + 1
            if (outcome == OK and grid is not None and action is GameAction.ACTION7
                    and self.keys_back[0] is not None and self.mask is not None):
                aid = action.value
                self.undo_tries[aid] = self.undo_tries.get(aid, 0) + 1
                if fp.state_key(grid, self.mask) == self.keys_back[0]:
                    self.undo_hits[aid] = self.undo_hits.get(aid, 0) + 1
            self.last_cand = None
            if outcome == GAME_OVER:
                if self.dyn.last_death_budget:
                    # The move budget ran out: the edge did nothing wrong, leave it untested.
                    self.last_key = self.last_idx = None
                    self.stats["budget_deaths"] += 1
                elif me is not None and not action.is_complex():
                    self.fatal.add((me.x1, me.y1, action.value))
                    self.stats["fatal_moves"] += 1
            if outcome != OK:
                after = None
        self.cur_grid = grid
        # Within a level the mask is fixed, so this frame's objects are the ones just extracted.
        self._cur_objs = after
        self._verdicts = {}
        self.available = [int(getattr(a, "value", a)) for a in (obs.available_actions or [])]

        action, data = super().act(obs)
        self.level_actions += 1

        if action is GameAction.RESET or grid is None:
            if obs.state != GameState.GAME_OVER:
                self.dyn.new_attempt()
            self.last_cand = None
            self._drop_plan()
            self.keys_back = [None, None]
        else:
            self.last_cand = (action, data)
            self.last_grid = grid
            self.last_objs = self.cur_objs()
            key = fp.state_key(grid, self.mask) if self.mask is not None else None
            self.keys_back = [self.keys_back[1], key]
        return action, data

    def _choose(self, node: Node) -> int | None:
        if self.advisor is not None and self.cur_grid is not None and self._advice_due():
            self._ask_advisor(node)
        if self.cur_grid is None or not self.dyn.observations:
            return super()._choose(node)
        objs = self.cur_objs()
        avatar = self.dyn.avatar()
        me = self.dyn.body(find(objs, avatar), objs) if avatar is not None else None
        if node.summary is None:
            node.summary = self._summary(objs)
        if node.h is None and (self.relations or self.confirmed or self.steer_first or self.experiment):
            if self.experiment:
                # each node's readings are kept, so that its distance can be measured again
                # against the next hypothesis tested
                self.node_rel[node.key] = self._rel_heuristic(objs)
                if self.hyp_values is not None:
                    self.node_values[node.key] = dict(self.hyp_values)
            node.h = self._heuristic(objs)
            if node.h is not None:
                self.best_first = True
        self.last_h = node.h
        idx = self._replay_step(node)
        if idx is not None:
            return idx
        if self.imagine_clicks and self.track_hypotheses:
            idx = self._imagined_click(node, objs)
            if idx is not None:
                return idx
        if self.planning:
            idx = self._follow_plan(node, me)
            if idx is None and not self.plan:
                idx = self._make_plan(node, me, "goal")
            if idx is None and not self.plan and self.advice_targets:
                idx = self._make_plan(node, me, "advice")
            if idx is not None:
                if self.plan_kind != "refill" and self._short_of_energy(len(self.plan)):
                    # The resource will run out before the plan ends: go and refill it
                    # if a tile for that is in reach, else carry on and take the loss.
                    saved = (self.plan, self.plan_expect, self.plan_kind)
                    self._drop_plan()
                    refill = self._refill_plan(node, me)
                    if refill is not None:
                        return refill
                    self.plan, self.plan_expect, self.plan_kind = saved
                elif self.plan_kind != "refill" and self._beyond_budget(len(self.plan)):
                    # The plan cannot finish in this attempt of the move budget: a reset
                    # costs one action and starts a fresh budget, the walk would cost
                    # them all.
                    self._drop_plan()
                    self.stats["budget_resets"] += 1
                    return None
                return idx
        for i in node.untested(self.graph.max_group(), self.graph.epoch):
            c = node.cands[i]
            if self.skip_undo and c.data is None and self._is_undo(c.action.value):
                if self.graph.epoch == 0:
                    # it only takes the game back a step, to a frame already seen
                    self.graph.mark_tested(node.key, i)
                    self.stats["undo_skipped"] = self.stats.get("undo_skipped", 0) + 1
                    continue
            if me is not None and c.data is None and (me.x1, me.y1, c.action.value) in self.fatal:
                verdict = "danger"  # this very move lost the game in this level
            else:
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
        if self._short_of_energy(2) and not self.plan:
            # The resource is nearly out: refill while a tile is still in reach.
            idx = self._refill_plan(node, me)
            if idx is not None:
                return idx
        if self._frontier_beyond_budget(node):
            self.stats["budget_resets"] += 1
            return None
        return super()._choose(node)

    def _is_undo(self, aid: int) -> bool:
        """The undo key: every time it was tried after another action (at least twice), it gave
        back the frame from before that action."""
        if aid != GameAction.ACTION7.value:
            return False
        tries = self.undo_tries.get(aid, 0)
        return tries >= 2 and self.undo_hits.get(aid, 0) == tries

    # -- goal hypotheses and the memory of causes (Phase 59) -----------------------

    def _me(self, objs: list[Obj]) -> Obj | None:
        avatar = self.dyn.avatar()
        return self.dyn.body(find(objs, avatar), objs) if avatar is not None else None

    def _measures(self, objs: list[Obj], grid: np.ndarray) -> dict[tuple, int | None]:
        me = self._me(objs)
        return {h.key: hy.measure(h, objs, grid, me) for h in self.hyps or []}

    def _track_hypotheses(self, action, data, outcome: str, after: list[Obj] | None,
                          grid: np.ndarray | None) -> None:
        """Credit the last action with what it did to every hypothesis's measure; at a level's
        end, keep the hypotheses' readings on its last frame."""
        if outcome == LEVEL_UP:
            if self.hyps and self.hyp_values is not None:
                self.hyp_reports.append([
                    (h.kind, h.params, h.first, h.best, self.hyp_values.get(h.key), h.goal, self.causes.live(h.key))
                    for h in self.hyps])
                for h in self.hyps:
                    last = self.hyp_values.get(h.key)
                    if self._explains(h, last):
                        t = self._hyp_type(h)
                        if t in self.confirmed:
                            self.confirmed.remove(t)
                        self.confirmed.insert(0, t)
                        self.confirm_count[t] += 1
                        self.stats["hyp_confirmed"] = self.stats.get("hyp_confirmed", 0) + 1
            self.hyps = None
            self.hyp_values = None
            return
        if outcome != OK or after is None or grid is None:
            self.hyp_values = None  # a restart: measured again on the next frame
            return
        if self.hyps is None:
            self.hyps = hy.propose(after, grid, self.dyn.avatar())
            self.hyp_values = None
        values = self._measures(after, grid)
        if self.hyp_values is not None:
            if data is not None:
                target = at(self.last_objs, data["x"], data["y"]) if self.last_objs else None
                cls: object = ("click", target.esig if target is not None else None)
            else:
                cls = action.value
            for h in self.hyps:
                self.causes.record(h.key, cls, self.hyp_values.get(h.key), values.get(h.key),
                                   self._hyp_type(h) if self.carry_causes else None)
        for h in self.hyps:
            v = values.get(h.key)
            if v is not None:
                if h.first is None:
                    h.first = v
                h.best = v if h.best is None else min(h.best, v)
        self.hyp_values = values
        if self.experiment:
            self._experiment(values)

    @staticmethod
    def _explains(h: hy.Hypothesis, last: int | None) -> bool:
        """The hypothesis moved from where the level began to (or within a step of) its goal."""
        return (h.first is not None and last is not None and last < h.first
                and last - h.goal <= max(1, (h.first - h.goal) // 4))

    @staticmethod
    def _hyp_type(h: hy.Hypothesis) -> tuple:
        """A hypothesis up to recolouring and placement: what carries over to the next level."""
        if h.kind == "match":
            return ("match",)
        if h.kind == "unify":
            return ("unify", h.params[0])
        return (h.kind, h.params[0][1:])  # reach / clear: the kind's size and shape

    def _steering(self) -> list[hy.Hypothesis]:
        """This level's hypotheses to steer by: those of a kind that ended an earlier level, else
        (steer_first) those some action has moved closer and that are not met yet."""
        if not self.hyps or self.hyp_values is None:
            return []
        out = []
        if self.steer_confirmed and self.confirmed:
            present = {self._hyp_type(h) for h in self.hyps} & set(self.confirmed)
            if self.narrow_confirmed and present:
                # the kinds that ended the most levels: a kind that ended one level along with the
                # true goal does not dilute the steering once the goal has ended two
                top = max(self.confirm_count[t] for t in present)
                present = {t for t in present if self.confirm_count[t] == top}
            out = [h for h in self.hyps if self._hyp_type(h) in present]
        if not out and self.experiment and self.trial is not None:
            out = [self.trial]
        if not out and self.steer_first and not self.confirmed and self.level_actions >= self.steer_first_after:
            out = [h for h in self.hyps if self.causes.helped(h.key)
                   and (self.hyp_values.get(h.key) or 0) > h.goal]
        return out

    def _hyp_heuristic(self, values: dict | None = None) -> float | None:
        """The steering hypotheses' distance from their goals, each as a share of where it began."""
        steer = self._steering()
        values = self.hyp_values if values is None else values
        if not steer or values is None:
            return None
        total, found = 0.0, False
        for h in steer:
            v = values.get(h.key)
            if v is None or h.first is None:
                continue
            total += max(0, v - h.goal) / max(1, h.first - h.goal)
            found = True
        return total if found else None

    def _steer_candidates(self, node: Node, pool: list[int]) -> list[int]:
        """Candidates whose action class has, on average, moved a steering hypothesis closer."""
        steer = self._steering()
        if not steer:
            return []
        objs = self.cur_objs()
        best, out = 0.0, []
        for i in pool:
            c = node.cands[i]
            if c.data is not None:
                t = at(objs, c.data["x"], c.data["y"])
                cls: object = ("click", t.esig if t is not None else None)
            else:
                cls = c.action.value
            g = max((self.causes.mean_gain(h.key, cls, self._hyp_type(h) if self.carry_causes else None)
                     for h in steer), default=0.0)
            if g > best:
                best, out = g, [i]
            elif g == best and g > 0:
                out.append(i)
        return out

    # -- goal experiments (Phase 61) ---------------------------------------------

    def _confirmed_present(self) -> bool:
        return bool(self.steer_confirmed and self.confirmed and self.hyps
                    and {self._hyp_type(h) for h in self.hyps} & set(self.confirmed))

    def _experiment(self, values: dict) -> None:
        """Test one hypothesis at a time: steer by it alone until its measure reaches its goal
        (and the level did not end: not the goal, or not all of it), or until it has not come
        closer for ``experiment_patience`` actions; then take the next one some action has moved
        closer. A hypothesis that ends the level is confirmed as in Phase 59."""
        if self.trial is not None:
            v = values.get(self.trial.key)
            if v is not None and v <= self.trial.goal:
                self._end_trial("trials_met")
            elif v is not None and (self.trial_best is None or v < self.trial_best):
                self.trial_best = v
                self.trial_quiet = 0
            else:
                self.trial_quiet += 1
                if self.trial_quiet >= self.experiment_patience:
                    self._end_trial("trials_stuck")
        if self.trial is None and self.hyps and self.level_actions >= self.experiment_after \
                and not self._confirmed_present():
            order = {"reach": 0, "match": 1, "clear": 2, "unify": 3}
            pool = [h for h in self.hyps if h.key not in self.tried and self.causes.helped(h.key)
                    and values.get(h.key) is not None and values[h.key] > h.goal]
            if pool:
                def rank(h: hy.Hypothesis) -> tuple:
                    v = values[h.key]
                    share = (v - h.goal) / max(1, (h.first if h.first is not None else v) - h.goal)
                    return (share, order.get(h.kind, 9), repr(h.params))
                self.trial = min(pool, key=rank)
                self.trial_best = values[self.trial.key]
                self.trial_quiet = 0
                self.stats["trials"] = self.stats.get("trials", 0) + 1
                self._remeasure()

    def _end_trial(self, why: str) -> None:
        assert self.trial is not None
        self.tried.add(self.trial.key)
        self.trial = None
        self.trial_best = None
        self.stats[why] = self.stats.get(why, 0) + 1
        self._remeasure()

    def _remeasure(self) -> None:
        """The steering changed: every known node's distance is measured again."""
        for key, node in self.graph.nodes.items():
            vals = self.node_values.get(key)
            node.h = self._combine(self.node_rel.get(key), self._hyp_heuristic(vals) if vals is not None else None)
        self.graph._dist_cache.clear()

    # -- clicks planned in imagination (Phase 60) ---------------------------------

    def _imagined_click(self, node: Node, objs: list[Obj]) -> int | None:
        """The next click of a plan made in imagination towards a confirmed goal; a new plan when
        none is under way and the level has a confirmed goal and clicks of known effect."""
        if self.click_plan and self.click_expect is not None:
            mask = self.mask
            same = (np.array_equal(np.where(mask, 0, self.cur_grid), np.where(mask, 0, self.click_expect))
                    if mask is not None else np.array_equal(self.cur_grid, self.click_expect))
            if not same:
                self.click_plan = []
                self.stats["imagined_misses"] = self.stats.get("imagined_misses", 0) + 1
        self.click_expect = None
        if not self.click_plan:
            if self.imagined_this_level >= 8 or self.cur_grid is None:
                return None
            # a search is worth making again only when the model of clicks has learned something
            known = self.effects.known_count()
            if known == 0 or known <= self.imagined_known:
                return None
            steer = self._steering()
            if not steer:
                return None
            self.imagined_known = known
            self.imagined_this_level += 1

            def measure(o: list[Obj], g: np.ndarray) -> int | None:
                me = self._me(o)
                total = 0
                for h in steer:
                    v = hy.measure(h, o, g, me)
                    if v is None:
                        return None
                    total += max(0, v - h.goal)
                return total

            plan = imagine(self.cur_grid, self.mask, self.effects, extract, measure)
            self.stats["imagined_searches"] = self.stats.get("imagined_searches", 0) + 1
            if not plan:
                return None
            self.stats["imagined_plans"] = self.stats.get("imagined_plans", 0) + 1
            self.click_plan = list(plan)
        target = self.click_plan.pop(0)
        here = self._locate(target, objs)
        if here is None:
            self.click_plan = []
            return None
        for i, c in enumerate(node.cands):
            if c.data is not None and here.contains(c.data["x"], c.data["y"]):
                self.click_expect = self.effects.predict(here, self.cur_grid, self.mask)
                self.stats["imagined_clicks"] = self.stats.get("imagined_clicks", 0) + 1
                return i
        self.click_plan = []
        return None

    # -- the goal advisor (Phase 58) ---------------------------------------------

    def _advice_due(self) -> bool:
        if self.advisor is None or not self.advisor.available:
            return False
        if self.advice_queries >= MAX_ADVICE_PER_LEVEL or self.stats["advice_queries"] >= MAX_ADVICE_PER_GAME:
            return False
        if self.advice_next < len(ADVICE_AT) and self.level_actions >= ADVICE_AT[self.advice_next]:
            return True
        # Everything advised was done and the level goes on: ask again, on the frame it led to.
        return (bool(self.advice_targets) and len(self.advice_done) >= len(self.advice_targets)
                and self.level_actions - self.advice_asked_at >= 2)

    def _ask_advisor(self, node: Node) -> None:
        """Show the advisor this frame, its objects and what the agent knows; keep its targets."""
        objs = self.cur_objs()
        clicks = [c.data for c in node.cands if c.data is not None]
        if clicks and not any(a in self.available for a in (1, 2, 3, 4, 5)):
            # only clicks: show the objects a click of the agent's can land on
            shown = listed_objects([o for o in objs if any(o.contains(d["x"], d["y"]) for d in clicks)])
        else:
            shown = listed_objects(objs)
        prompt = build_prompt(describe_objects(shown, objs), self.available, self._advice_notes(shown),
                              max(0, self.levels_completed))
        self.advice_queries += 1
        while self.advice_next < len(ADVICE_AT) and self.level_actions >= ADVICE_AT[self.advice_next]:
            self.advice_next += 1
        self.advice_asked_at = self.level_actions
        self.advice_targets = []
        self.stats["advice_queries"] += 1
        advice = self.advisor.advise(self.cur_grid, prompt, len(shown))
        if advice is None:
            return
        self.stats["advice_answers"] += 1
        self.advice_targets = [shown[t - 1] for t in advice.targets]
        self.advice_done = set()
        self.advice_current = None
        self.advice_log.append(f"L{max(0, self.levels_completed) + 1}@{self.level_actions}: {advice.goal} "
                               f"-> {advice.targets}")
        self.stats["advice_last"] = self.advice_log[-1][:200]

    def _advice_notes(self, shown: list[Obj]) -> list[str]:
        """What the agent has learned, in the advisor's terms (object numbers, colour names)."""
        from .advisor import COLOUR_NAMES

        def number(sig: tuple) -> str:
            for i, o in enumerate(shown, 1):
                if o.sig == sig:
                    return f"#{i}"
            return f"a {COLOUR_NAMES[sig[0]]} object"

        notes = []
        avatar = self.dyn.avatar()
        if avatar is not None:
            moves = []
            for aid in (1, 2, 3, 4, 5):
                d = self.dyn.direction(aid)
                if d is not None:
                    moves.append(f"ACTION{aid} moves it by ({d[0]:+d}, {d[1]:+d})")
            notes.append(f"The keys move {number(avatar)}" + (": " + ", ".join(moves) if moves else "") + ".")
            walls = [COLOUR_NAMES[c] for c in range(16) if self.dyn.blocks(c)]
            if walls:
                notes.append("It cannot pass through " + ", ".join(walls) + ".")
        dead = [sig for sig, c in self.dyn.click.items()
                if sum(c.values()) >= 3 and c["none"] == sum(c.values())]
        if dead:
            notes.append("Clicking these kinds changed nothing: "
                         + ", ".join(f"{COLOUR_NAMES[e[0]]} {e[2]}x{e[1]}" for e in dead[:8]) + ".")
        if self.dyn.level_ups:
            done = []
            for sig in self.dyn.goal_sigs():
                done.append(f"the player reached {number(sig)}")
            for c in self.dyn.goal_colors():
                done.append(f"the player entered {COLOUR_NAMES[c]}")
            for e in self.dyn.goal_click_sigs():
                done.append(f"a {COLOUR_NAMES[e[0]]} {e[2]}x{e[1]} object was clicked")
            if done:
                notes.append("Earlier levels ended when " + "; or ".join(done[:4]) + ".")
        if self.advice_tried:
            notes.append("Already tried in this level without ending it: " + "; ".join(self.advice_tried[-6:]) + ".")
        if self.level_actions:
            notes.append(f"{self.level_actions} actions have been spent on this level.")
        return notes

    @staticmethod
    def _label(o: Obj) -> str:
        from .advisor import COLOUR_NAMES

        return f"the {COLOUR_NAMES[o.color]} {o.w}x{o.h} object at x={o.x1}, y={o.y1}"

    def _locate(self, target: Obj, objs: list[Obj]) -> Obj | None:
        """The advised object in the current frame: same exact shape and colour, nearest."""
        same = [o for o in objs if o.esig == target.esig] or [o for o in objs if o.sig == target.sig]
        if not same:
            return None
        return min(same, key=lambda o: abs(o.x1 - target.x1) + abs(o.y1 - target.y1))

    def _advice_mask(self, me: Obj) -> np.ndarray | None:
        """Cells of the first advised object not yet reached, other than the player."""
        objs = self.cur_objs()
        for i, t in enumerate(self.advice_targets):
            if i in self.advice_done:
                continue
            o = self._locate(t, objs)
            if o is None or o.sig == me.sig or self._touching(me, o):
                self.advice_done.add(i)
                continue
            mask = np.zeros(self.cur_grid.shape, dtype=bool)
            mask[o.pixels[:, 0], o.pixels[:, 1]] = True
            self.advice_current = i
            return mask
        return None

    def _advised_click(self, node: Node, local: list[int]) -> int | None:
        """An untested click on the first advised object that has one here, in whatever
        priority group the explorer put it."""
        if not self.advice_targets:
            return None
        objs = self.cur_objs()
        local = node.untested(self.graph.max_group(), self.graph.epoch)
        for i, t in enumerate(self.advice_targets):
            if i in self.advice_done:
                continue
            o = self._locate(t, objs)
            if o is None:
                self.advice_done.add(i)
                continue
            for j in local:
                c = node.cands[j]
                if c.data is not None and o.contains(c.data["x"], c.data["y"]):
                    self.advice_done.add(i)
                    self.advice_tried.append("clicked " + self._label(o))
                    return j
        return None

    # -- move budgets ---------------------------------------------------------

    def _remaining(self) -> int | None:
        """Actions left in this attempt under the level's move budget, if known
        and the attempt is under way."""
        budget = self.dyn.budget()
        if budget is None or self.since_reset == 0:
            return None
        return budget - self.dyn.attempt

    def _short_of_energy(self, actions: int) -> bool:
        """The resource bar pays for fewer actions than ``actions``."""
        if self.cur_grid is None:
            return False
        energy = self.dyn.energy_left(self.cur_grid, self.mask)
        return energy is not None and actions > energy

    def _refill_plan(self, node: Node, me: Obj | None) -> int | None:
        """Out of steps for what comes next: walk to a tile that refills the resource,
        if one is on screen and within the energy left."""
        if me is None or self.cur_grid is None or self.plan:
            return None
        energy = self.dyn.energy_left(self.cur_grid, self.mask)
        tiles = self.dyn.refill_tiles()
        if energy is None or not tiles:
            return None
        objs = self.cur_objs()
        target = np.zeros(self.cur_grid.shape, dtype=bool)
        for o in objs:
            if o.sig in tiles:
                target[o.pixels[:, 0], o.pixels[:, 1]] = True
        if not target.any():
            return None
        dirs = self.dyn.directions()
        if not dirs:
            return None
        static, zones = self.dyn.hazard_zones(objs, self.cur_grid, self.cur_grid.shape, HORIZON)
        found = plan_path(
            self.cur_grid, me, dirs, self.dyn.passable,
            lambda cols, coords: any(target[y, x] for y, x in coords),
            avoid=static, avoid_t=zones,
            forbidden=lambda pos, aid: (pos[0], pos[1], aid) in self.fatal,
        )
        if found is None or len(found[0]) > energy:
            return None
        self.plan, self.plan_expect = found
        self.plan_kind = "refill"
        self.stats["refill_plans"] += 1
        return self._plan_step(node)

    def _beyond_budget(self, actions: int) -> bool:
        remaining = self._remaining()
        return remaining is not None and actions > remaining

    def _frontier_beyond_budget(self, node: Node) -> bool:
        """Nothing to test here, and the nearest untested edge is farther than the
        attempt has actions left: walking there would end in the budget."""
        remaining = self._remaining()
        if remaining is None or node.untested(self.graph.max_group(), self.graph.epoch):
            return False
        dist, _ = self.graph.distances(self.graph.max_group())
        here = dist.get(node.key)
        if here is None or here + 1 <= remaining:
            return False
        # Only worth it when a fresh attempt can reach the frontier from the start;
        # otherwise resetting on every visit would never get anywhere.
        budget = self.dyn.budget()
        start = dist.get(self.first_key) if self.first_key is not None else None
        return budget is not None and start is not None and start + 1 <= budget

    # -- planning -------------------------------------------------------------

    def _follow_plan(self, node: Node, me: Obj | None) -> int | None:
        """Next step of the current plan, after checking the last one landed as predicted."""
        if not self.plan:
            return None
        expect = self.plan_expect[0]
        # (-1, -1) marks the final move, which is not held to a landing position:
        # whether the goal admitted the avatar or not, the plan is done.
        if expect != (-1, -1) and (me is None or (me.x1, me.y1) != expect):
            self._drop_plan()
            self.plan_cooldown = PLAN_COOLDOWN
            self.stats["plan_failures"] += 1
            return None
        self.plan.pop(0)
        self.plan_expect.pop(0)
        if not self.plan:
            self._plan_finished()
            return None
        return self._plan_step(node)

    def _plan_finished(self) -> None:
        """The plan's last move was executed and the level did not end."""
        objs = self.cur_objs()
        if self.plan_kind == "advice" and self.advice_current is not None:
            self.advice_done.add(self.advice_current)
            self.advice_tried.append("reached " + self._label(self.advice_targets[self.advice_current]))
            self.advice_current = None
        if self.plan_kind == "goal" and self.dyn.switches():
            # The goal was reached in this avatar state and nothing happened: it is
            # locked until a switch changes the state.
            self.locked.add(self.dyn.indicator_state(objs))
            self.stats["locked_goals"] += 1
        elif self.plan_kind == "switch" and self.switch_from is not None:
            self.tried_switch.add(self.switch_from)
            self.switch_from = None

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
        self.stats[{"explore": "explore_steps", "advice": "advice_steps"}.get(self.plan_kind, "plan_steps")] += 1
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
            state = self.dyn.indicator_state(self.cur_objs())
            if state in self.locked:
                return self._make_switch_plan(node, me, state)
            if self.goal_plans_by_state.get(state, 0) >= MAX_PLANS_PER_LEVEL:
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
            tests = []
            if target.any():
                tests.append(lambda cols, coords: any(target[y, x] for y, x in coords))
                # When no route reaches the goal sprite, aim for whatever encloses it:
                # ls20's exit pattern sits inside a box that admits the avatar only in
                # the right state, and bumping the box is how the state gets tested.
                box = self._containers(goal_sigs)
                if box.any():
                    tests.append(lambda cols, coords: any(box[y, x] for y, x in coords))
            # Colour goals last, leaving out colours that are walls elsewhere.
            goals = {c for c in self.dyn.goal_colors() if not self.dyn.blocks(c)}
            if goals:
                tests.append(lambda cols, coords: bool(cols & goals))
            if not tests:
                return None
        elif kind == "advice":
            if self.advice_plans >= MAX_ADVICE_PLANS_PER_LEVEL:
                return None
            target = self._advice_mask(me)
            if target is None:
                return None
            tests = [lambda cols, coords: any(target[y, x] for y, x in coords)]
        else:
            if self.explore_plans_this_level >= MAX_EXPLORE_PLANS_PER_LEVEL:
                return None
            target = np.zeros(self.cur_grid.shape, dtype=bool)
            for o in self.cur_objs():
                if not o.scenery and o.sig != me.sig and o.sig not in self.dyn.touched:
                    target[o.pixels[:, 0], o.pixels[:, 1]] = True
            if not target.any():
                return None
            tests = [lambda cols, coords: any(target[y, x] for y, x in coords)]
        found = None
        static, zones = self.dyn.hazard_zones(self.cur_objs(), self.cur_grid, self.cur_grid.shape, HORIZON)
        for is_goal in tests:
            found = plan_path(
                self.cur_grid, me, dirs, self.dyn.passable, is_goal,
                avoid=static, avoid_t=zones,
                forbidden=lambda pos, aid: (pos[0], pos[1], aid) in self.fatal,
            )
            if found is not None:
                break
        budget = self.dyn.budget()
        if found is not None and budget is not None and len(found[0]) > budget - 1:
            found = None  # longer than any attempt can be
        if found is None:
            if kind == "advice":
                # no known route to this advised object: try the next one
                self.advice_done.add(self.advice_current)
                self.advice_current = None
                return None
            # No route through cells known to be passable yet; try again after a
            # cooldown, and do not spend the per-level allowance on the search.
            self.plan_cooldown = PLAN_COOLDOWN
            return None
        if kind == "goal":
            self.plans_this_level += 1
            self.goal_plans_by_state[state] = self.goal_plans_by_state.get(state, 0) + 1
        elif kind == "advice":
            self.advice_plans += 1
        else:
            self.explore_plans_this_level += 1
        self.plan, self.plan_expect = found
        self.plan_kind = kind
        self.stats[{"goal": "plans", "advice": "advice_plans"}.get(kind, "explore_plans")] += 1
        return self._plan_step(node)

    def _make_switch_plan(self, node: Node, me: Obj, state: frozenset) -> int | None:
        """The goal is locked in this avatar state: go and stand on a switch tile.
        Switches not yet tried from this state come first; a switch whose outcome
        from here is known to be another locked state is skipped."""
        if self.switch_plans_this_level >= MAX_SWITCH_PLANS_PER_LEVEL or self.cur_grid is None:
            return None
        switches = self.dyn.switches()
        objs = self.cur_objs()
        present = [t for t in switches if any(o.sig == t for o in objs)]
        untried = [t for t in present if (state, t) not in self.tried_switch]
        useful = [t for t in present if self.dyn.state_graph.get((state, t)) not in self.locked]
        choice = untried or useful
        if not choice:
            return None
        self.rng.shuffle(choice)
        target = np.zeros(self.cur_grid.shape, dtype=bool)
        for o in objs:
            if o.sig in choice:
                target[o.pixels[:, 0], o.pixels[:, 1]] = True
        static, zones = self.dyn.hazard_zones(objs, self.cur_grid, self.cur_grid.shape, HORIZON)
        found = plan_path(
            self.cur_grid, me, self.dyn.directions(), self.dyn.passable,
            lambda cols, coords: any(target[y, x] for y, x in coords),
            avoid=static, avoid_t=zones,
            forbidden=lambda pos, aid: (pos[0], pos[1], aid) in self.fatal,
        )
        if found is None:
            self.plan_cooldown = PLAN_COOLDOWN
            return None
        self.switch_plans_this_level += 1
        self.plan, self.plan_expect = found
        self.plan_kind = "switch"
        last = self.plan_expect[-1]
        hit = next((o.sig for o in objs if o.sig in choice and o.x1 <= last[0] + me.w and o.x2 >= last[0] - me.w and o.y1 <= last[1] + me.h and o.y2 >= last[1] - me.h), choice[0])
        self.switch_from = (state, hit)
        self.stats["switch_plans"] += 1
        return self._plan_step(node)

    def _containers(self, goal_sigs: set[tuple]) -> np.ndarray:
        """Pixels of the sprites whose bounding box encloses a goal sprite."""
        assert self.cur_grid is not None
        objs = self.cur_objs()
        box = np.zeros(self.cur_grid.shape, dtype=bool)
        for g in objs:
            if g.sig not in goal_sigs:
                continue
            for o in objs:
                if o is g or o.scenery or o.sig in goal_sigs:
                    continue
                if o.x1 <= g.x1 and o.y1 <= g.y1 and o.x2 >= g.x2 and o.y2 >= g.y2:
                    box[o.pixels[:, 0], o.pixels[:, 1]] = True
        return box

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
        me = self.dyn.body(find(objs, avatar), objs) if avatar is not None else None
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
        advised = self._advised_click(node, local)
        if advised is not None:
            self.stats["advice_clicks"] += 1
            return advised
        if self.track_hypotheses:
            steer = self._steer_candidates(node, [i for i in local if self._verdicts.get(i, "unknown") == "unknown"] or local)
            if steer:
                self.stats["hyp_steps"] = self.stats.get("hyp_steps", 0) + 1
                return self.rng.choice(steer)
        unknown = [i for i in local if self._verdicts.get(i, "unknown") == "unknown"]
        pool = unknown or local
        if self.relations and self.kind_tries:
            # Clicks on kinds that have closed the goal gap before go first; kinds that
            # have been tried enough and never closed it go last.
            objs = self.cur_objs()

            def mean_gain(i: int) -> float | None:
                c = node.cands[i]
                if c.data is None:
                    return None
                t = at(objs, c.data["x"], c.data["y"])
                kind = self._kind_of(t, self.kind_tries) if t is not None else None
                if kind is None:
                    return None
                return self.kind_gain[kind] / self.kind_tries[kind]

            gains = {i: mean_gain(i) for i in pool}
            helpful = [i for i, g in gains.items() if g is not None and g < 0]
            if helpful:
                best = min(gains[i] for i in helpful)
                self.stats["gain_clicks"] += 1
                return self.rng.choice([i for i in helpful if gains[i] <= best])
            fresh = [i for i, g in gains.items() if g is None or self.kind_tries.get(0, 0) == 0 and g == 0]
            if fresh and len(fresh) < len(pool):
                pool = fresh
        if self.relations and self.near_clicks:
            near = self._near_moving(node, pool, self.cur_objs())
            if near:
                self.stats["near_clicks"] += 1
                return self.rng.choice(near)
        if unknown:
            self.stats["ran_unknown"] += 1
            return self.rng.choice(pool if set(pool) <= set(unknown) else unknown)
        self.stats["ran_predicted"] += 1
        return self.rng.choice(pool)


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
