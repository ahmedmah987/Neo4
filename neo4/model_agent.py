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
MAX_PLANS_PER_LEVEL = 5  # goal plans per level and avatar state
MAX_EXPLORE_PLANS_PER_LEVEL = 40
MAX_SWITCH_PLANS_PER_LEVEL = 30
PLAN_COOLDOWN = 20  # exploration steps after a failed plan before planning again
HORIZON = 40  # moves ahead over which a patrol's motion is extrapolated for planning


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
        self.stats.update(
            {"skipped": 0, "ran_unknown": 0, "ran_predicted": 0,
             "plans": 0, "plan_steps": 0, "plan_failures": 0, "goal_clicks": 0,
             "explore_plans": 0, "explore_steps": 0,
             "noise_sigs": 0, "graph_rebuilds": 0, "budget_deaths": 0, "fatal_moves": 0,
             "switch_plans": 0, "locked_goals": 0, "budget_resets": 0, "refill_plans": 0,
             "descents": 0, "near_clicks": 0}
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
        self.fatal = set()
        self.locked = set()
        self.tried_switch = set()
        self.goal_plans_by_state = {}
        self.switch_plans_this_level = 0
        self.switch_from = None
        self._cur_objs = None  # the mask changed with the level
        self.level_start_objs = list(self.cur_objs()) if self.cur_grid is not None else None
        self.best_first = bool(self.relations)

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
        """From a level's first and last frames: the kinds whose instances moved (or
        appeared) and ended nearest to a kind that stood still. Candidates are ranked
        by how steadily their gap closed along the solution path (the shortest known
        route through the level graph), then by how much it closed; the two best
        pairs become the heuristic for later levels."""
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
        if scored:
            self.relations = [(m, t) for _, _, _, m, t in scored[:2]]
            self.best_first = True

    def _heuristic(self, objs: list[Obj]) -> float | None:
        """Sum over the learned relations of the gap between the nearest M and T."""
        if not self.relations:
            return None
        total = 0
        found = False
        for m_sig, t_sig in self.relations:
            ms = [o for o in objs if o.sig == m_sig]
            ts = [o for o in objs if o.sig == t_sig]
            if ms and ts:
                total += min(self._gap(a, b) for a in ms for b in ts)
                found = True
        return float(total) if found else None

    def _near_moving(self, node: Node, local: list[int], objs: list[Obj]) -> list[int]:
        """Click candidates nearest to an instance of a moving kind of the relation."""
        ms = [o for m_sig, _ in self.relations for o in objs if o.sig == m_sig]
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
            if outcome == LEVEL_UP:
                self._learn_relations(self.level_start_objs, self.last_objs)
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

        action, data = super().act(obs)

        if action is GameAction.RESET or grid is None:
            if obs.state != GameState.GAME_OVER:
                self.dyn.new_attempt()
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
        me = self.dyn.body(find(objs, avatar), objs) if avatar is not None else None
        if node.summary is None:
            node.summary = self._summary(objs)
        if node.h is None and self.relations:
            node.h = self._heuristic(objs)
        idx = self._replay_step(node)
        if idx is not None:
            return idx
        if self.planning:
            idx = self._follow_plan(node, me)
            if idx is None and not self.plan:
                idx = self._make_plan(node, me, "goal")
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
        self.stats["plan_steps" if self.plan_kind != "explore" else "explore_steps"] += 1
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
            # No route through cells known to be passable yet; try again after a
            # cooldown, and do not spend the per-level allowance on the search.
            self.plan_cooldown = PLAN_COOLDOWN
            return None
        if kind == "goal":
            self.plans_this_level += 1
            self.goal_plans_by_state[state] = self.goal_plans_by_state.get(state, 0) + 1
        else:
            self.explore_plans_this_level += 1
        self.plan, self.plan_expect = found
        self.plan_kind = kind
        self.stats["plans" if kind == "goal" else "explore_plans"] += 1
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
        unknown = [i for i in local if self._verdicts.get(i, "unknown") == "unknown"]
        pool = unknown or local
        if self.relations:
            near = self._near_moving(node, pool, self.cur_objs())
            if near:
                self.stats["near_clicks"] += 1
                return self.rng.choice(near)
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
