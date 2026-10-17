"""Rule induction: what each action does, learned from recorded transitions.

``Dynamics`` keeps per-game statistics that survive level changes:

* how far each object signature moves under each keyboard action,
* which object is the *avatar* (the one the keyboard moves),
* for each colour, how often it blocked the avatar versus let it pass (the
  avatar's move is a *sweep*: every cell between its position and its
  destination counts as crossed, and the nearest band counts as blocking),
* which colours were in front of the avatar when the game was lost, and which
  object kinds were touching the avatar's place when it was lost (a patrol that
  kills on contact),
* whether the game ends every attempt after a fixed number of actions (a move
  budget), in which case those losses are not blamed on anything,
* which tile kinds are *switches*: entering them changes something far from the
  avatar (ls20's floor tiles cycle a pattern shown in a panel); the far objects
  they change are *indicators*, and their arrangement is the avatar's state,
* how often clicking a given object signature changed anything.

``predict`` turns those statistics into a verdict on a candidate action in a
given frame — ``unknown``, ``move``, ``blocked``, ``danger`` or
``noop_click`` — with the evidence count behind it, and ``describe`` renders
the learned rules in English.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from arcengine import GameAction

from .objects import Diff, Obj, at, diff, find
from .planner import swept

OK, GAME_OVER, LEVEL_UP = "ok", "game_over", "level_up"
BUDGET = "budget"  # a loss that the attempt's move budget explains, not the move

MIN_SUPPORT = 3  # observations before a rule may be used for a prediction
BLOCK_RATE = 0.9  # a colour that stopped the avatar at least this often blocks
PASS_RATE = 0.9  # a colour the avatar crossed at least this often is passable
DANGER_RATE = 0.5  # share of visits that ended the game
DEATH_SUPPORT = 2  # lost games on a click target before its clicks count as dangerous
NOOP_RATE = 0.95  # share of clicks on a signature that changed nothing
DIR_PURITY = 0.7  # share of an action's observed moves that agree on one direction
GOAL_LIFT = 5.0  # a goal feature ends levels at least this many times the base rate
NOISE_RATE = 0.5  # a signature churning on this share of steps under every action is noise
NOISE_MIN_STEPS = 10  # observations an action class needs before it counts for noise
NOISE_REFRESH = 25  # recompute the noise set every this many observations
MIN_BUDGET = 5  # an attempt length below this is never read as a move budget
REMOTE_RATE = 0.9  # share of entries of a tile kind on which the same far object changed
FAR = 6  # cells beyond the avatar's body from which a change counts as remote


@dataclass(slots=True)
class Prediction:
    verdict: str  # unknown | move | blocked | danger | noop_click
    support: int = 0
    why: str = ""


class Dynamics:
    def __init__(self) -> None:
        self.move: dict[tuple[int, tuple], Counter] = defaultdict(Counter)
        self.click: dict[tuple, Counter] = defaultdict(Counter)
        self.cell: dict[int, Counter] = defaultdict(Counter)
        self.hazard: Counter = Counter()
        self.goal_ahead: Counter = Counter()  # colours ahead of the avatar when a level ended
        self.goal_click: Counter = Counter()  # signatures clicked when a level ended
        self.goal_click_color: Counter = Counter()  # colours clicked when a level ended
        self.click_color: Counter = Counter()  # colours clicked, all outcomes
        self.level_ups = 0
        self.click_death: Counter = Counter()  # signatures clicked when the game was lost
        # The same, per object instance (signature + position): outcomes often depend on
        # *which* of several identical objects is clicked, not on their kind.
        self.click_at: dict[tuple, Counter] = defaultdict(Counter)
        self.click_death_at: Counter = Counter()
        self.churn: dict[tuple, Counter] = defaultdict(Counter)  # sig -> action class -> events
        self.touched: set[tuple] = set()  # object signatures the avatar has moved into / against
        self.touch_count: Counter = Counter()  # signature -> times it was ahead of the avatar
        self.goal_ahead_sig: Counter = Counter()  # object signatures entered when a level ended
        self.goal_adjacent: Counter = Counter()  # signatures touching the avatar when a level ended
        self.adjacent_count: Counter = Counter()  # signature -> keyboard steps it touched the avatar
        self.goal_action: Counter = Counter()  # keyboard action id -> levels it completed
        self.instances: Counter = Counter()  # signature -> objects of it seen over keyboard frames
        self.key_frames = 0
        self.key_uses: Counter = Counter()  # keyboard action id -> times used
        self.key_death: Counter = Counter()  # keyboard action id -> times it lost the game
        self.steps_by_class: Counter = Counter()  # action class -> transitions observed
        self.observations = 0
        # sig -> {"with": moved by the avatar's vector while touching it, "without": did not}
        self.companion: dict[tuple, Counter] = defaultdict(Counter)
        self.killer: Counter = Counter()  # sig -> losses with it touching the avatar's place
        self.near_count: Counter = Counter()  # sig -> keyboard steps it ended beside the avatar
        self.contact_deaths = 0  # keyboard losses with a frame to look at, budget losses aside
        self.remote: dict[tuple, Counter] = defaultdict(Counter)  # tile sig -> far change -> count
        self.tile_entered: Counter = Counter()  # tile sig -> moves that ended on it
        self.state_graph: dict[tuple[frozenset, tuple], frozenset] = {}  # (state, switch) -> state
        self._indicators: set[tuple] = set()
        self.attempt = 0  # actions since the attempt began (level start, reset or loss)
        self.longest_attempt: Counter = Counter()  # level -> longest attempt seen there
        self.death_lengths: dict[int, Counter] = defaultdict(Counter)  # level -> length -> losses
        # level -> length -> distinct (avatar position, action) the losses came from: a
        # budget ends attempts wherever the avatar is; one fatal move repeated does not.
        self.death_places: dict[tuple[int, int], set] = defaultdict(set)
        self.last_death_budget = False  # the loss just observed was a budget loss
        self._avatar: tuple | None = None
        self._avatar_dirty = True
        self._noise: frozenset = frozenset()
        self._noise_at = 0

    # -- learning -------------------------------------------------------------

    def observe(
        self,
        action: GameAction,
        data: dict[str, int] | None,
        before: list[Obj],
        before_grid: np.ndarray,
        after: list[Obj] | None,
        outcome: str,
    ) -> None:
        self.observations += 1
        self.attempt += 1
        level = self.level_ups
        self.longest_attempt[level] = max(self.longest_attempt[level], self.attempt)
        self.last_death_budget = False
        if outcome == GAME_OVER:
            self.death_lengths[level][self.attempt] += 1
            place: tuple = (action.value, data["x"], data["y"]) if data else (action.value,)
            avatar = self.avatar()
            me = find(before, avatar) if avatar is not None else None
            if me is not None:
                place += (me.x1, me.y1)
            self.death_places[(level, self.attempt)].add(place)
            if self.attempt == self.budget():
                self.last_death_budget = True
                outcome = BUDGET
        if outcome != OK:
            self.attempt = 0
        if outcome == LEVEL_UP:
            self.level_ups += 1
        if not action.is_complex():
            self.key_frames += 1
            for sig, n in Counter(o.sig for o in before if not o.scenery).items():
                self.instances[sig] += n
        d = diff(before, after) if after is not None and outcome == OK else None
        if d is not None:
            # Clicks on different targets are different actions: something that
            # changes only when one particular target is clicked is a mechanic,
            # something that changes whatever is clicked is noise.
            if action.is_complex():
                target = at(before, data["x"], data["y"]) if data else None
                cls: object = ("click", target.esig if target is not None else None)
            else:
                cls = action.value
            self.steps_by_class[cls] += 1
            for b, _, _, _ in d.moved:
                self.churn[b.sig][cls] += 1
            for o in d.vanished:
                self.churn[o.sig][cls] += 1
            for o in d.appeared:
                self.churn[o.sig][cls] += 1
        if action.is_complex():
            self._observe_click(data, before, d, outcome)
        else:
            self._observe_key(action, before, before_grid, d, outcome, after)

    def new_attempt(self) -> None:
        """The agent reset the level: the move budget starts over."""
        self.attempt = 0

    def budget(self) -> int | None:
        """Actions an attempt may take in the current level before the game ends it,
        if the level has such a limit: at least two losses fell at the longest
        attempt length seen in it."""
        level = self.level_ups
        n = self.longest_attempt[level]
        if n < MIN_BUDGET or self.death_lengths[level][n] < 2:
            return None
        return n if len(self.death_places[(level, n)]) >= 2 else None

    def _observe_click(self, data, before, d: Diff | None, outcome: str) -> None:
        if data is None:
            return
        target = at(before, data["x"], data["y"])
        if target is None:
            return
        changed = outcome != OK or (d is not None and d.changed)
        key = "change" if changed else "none"
        self.click[target.esig][key] += 1
        self.click_at[(target.esig, target.x1, target.y1)][key] += 1
        self.click_color[target.color] += 1
        if outcome == LEVEL_UP:
            self.goal_click[target.esig] += 1
            self.goal_click_color[target.color] += 1
        elif outcome == GAME_OVER:
            self.click_death[target.esig] += 1
            self.click_death_at[(target.esig, target.x1, target.y1)] += 1

    def _observe_key(
        self, action, before, before_grid, d: Diff | None, outcome: str, after=None
    ) -> None:
        aid = action.value
        self.key_uses[aid] += 1
        if outcome == BUDGET:
            return  # the budget ran out on this move; the move itself is not to blame
        if outcome == GAME_OVER:
            self.key_death[aid] += 1
        if d is not None:
            for b, _, dx, dy in d.moved:
                self.move[(aid, b.sig)][(dx, dy)] += 1
                self._avatar_dirty = True
            for b in d.unchanged:
                if (aid, b.sig) in self.move:
                    self.move[(aid, b.sig)][(0, 0)] += 1
        avatar = self.avatar()
        if avatar is None:
            return
        me = find(before, avatar)
        if me is None:
            return
        if d is not None:
            self._observe_companions(me, before, d)
        # A sprite made of several colours (dc22's eye inside its body, ls20's bar
        # over its block) is handled as one body from here on.
        me = self.body(me, before)
        # Adjacency is recorded for every keyboard action, including ones that do
        # not move the avatar ("interact"): the thing beside the avatar may be what
        # the action acts on.
        adjacent = self._adjacent_sigs(me, before)
        for sig in adjacent:
            self.adjacent_count[sig] += 1
        if outcome == LEVEL_UP:
            self.goal_action[aid] += 1
            for sig in adjacent:
                self.goal_adjacent[sig] += 1
        direction = self.direction(aid)
        if after is not None:
            self._observe_contact(me, direction, before, after, outcome)
        if direction is None:
            return
        bands = self._swept(me, direction, before_grid)
        colors = set().union(*(c for c, _ in bands))
        coords = [cell for _, cells in bands for cell in cells]
        ahead_sigs = set()
        for y, x in coords:
            o = at(before, x, y)
            if o is not None and o.sig != avatar and not o.scenery:
                ahead_sigs.add(o.sig)
        for sig in ahead_sigs:
            self.touched.add(sig)
            self.touch_count[sig] += 1
        if outcome == GAME_OVER:
            for c in colors:
                self.hazard[c] += 1
            return
        if outcome == LEVEL_UP:
            for c in colors:
                self.goal_ahead[c] += 1
            for sig in ahead_sigs:
                self.goal_ahead_sig[sig] += 1
            return
        moved = d is not None and any(b.sig == avatar for b, _, _, _ in d.moved)
        if moved:
            for c in colors:
                self.cell[c]["passed"] += 1
            if after is not None:
                self._observe_remote(me, before, after, d)
        else:
            # Whatever stopped the avatar is somewhere in the sweep. Colours it is
            # known to cross are not to blame; the rest share the credit (early on
            # the floor too, until its passes establish it).
            suspects = [c for c in colors if not self._crosses(c)]
            for c in suspects or colors:
                self.cell[c]["blocked"] += 1

    def _observe_remote(self, me: Obj, before, after, d: Diff) -> None:
        """The tile kinds the avatar's body came to rest on, and what changed far
        away from it on the same step."""
        me2 = self.body(find(after, me.sig), after)
        if me2 is None:
            return
        comp = self.companions()
        foot = set(zip(me2.pixels[:, 0].tolist(), me2.pixels[:, 1].tolist()))
        tiles = set()
        for o in before:
            if o.scenery or o.sig == me.sig or o.sig in comp:
                continue
            if o.x1 > me2.x2 or o.x2 < me2.x1 or o.y1 > me2.y2 or o.y2 < me2.y1:
                continue
            if any((y, x) in foot for y, x in zip(o.pixels[:, 0].tolist(), o.pixels[:, 1].tolist())):
                tiles.add(o.sig)
        if not tiles:
            return

        def far(o: Obj) -> bool:
            return o.x2 < me2.x1 - FAR or o.x1 > me2.x2 + FAR or o.y2 < me2.y1 - FAR or o.y1 > me2.y2 + FAR

        changes: set[tuple] = set()
        for b, _, _, _ in d.moved:
            if far(b) and b.sig != me.sig and b.sig not in comp:
                changes.add(("moved", b.sig))
        for b, a in d.recolored:
            if far(b):
                changes.add(("recoloured", b.sig))
        for o in d.vanished:
            if far(o):
                changes.add(("vanished", o.sig))
        for o in d.appeared:
            if far(o):
                changes.add(("appeared", o.sig))
        s0 = self.indicator_state(before)
        for t in tiles:
            self.tile_entered[t] += 1
            for c in changes:
                self.remote[t][c] += 1
        switches = self.switches()
        if switches & tiles:
            s1 = self.indicator_state(after)
            if s1 != s0:
                for t in switches & tiles:
                    self.state_graph[(s0, t)] = s1

    def switches(self) -> set[tuple]:
        """Tile kinds whose entry changes the same far object nearly every time."""
        out = set()
        for t, n in self.tile_entered.items():
            if n >= MIN_SUPPORT and any(k / n >= REMOTE_RATE for k in self.remote[t].values()):
                out.add(t)
        return out

    def indicator_sigs(self) -> set[tuple]:
        """Object kinds that switches change: their arrangement is the avatar's state.
        Membership is sticky so that a state key does not change under the agent."""
        for t in self.switches():
            n = self.tile_entered[t]
            for (_, sig), k in self.remote[t].items():
                if k / n >= REMOTE_RATE:
                    self._indicators.add(sig)
        return self._indicators

    def indicator_state(self, objs) -> frozenset:
        sigs = self.indicator_sigs()
        if not sigs:
            return frozenset()
        return frozenset((o.esig, o.x1, o.y1) for o in objs if o.sig in sigs)

    def _observe_companions(self, me: Obj, before, d: Diff) -> None:
        """Which sprites touching the avatar moved by its vector on this step."""
        vec = next(((dx, dy) for b, _, dx, dy in d.moved if b.sig == me.sig), None)
        if vec is None or vec == (0, 0):
            return
        with_me = {id(b) for b, _, dx, dy in d.moved if (dx, dy) == vec}
        for o in before:
            if o is me or o.scenery or o.sig == me.sig or not self._touching(me, o):
                continue
            self.companion[o.sig]["with" if id(o) in with_me else "without"] += 1

    def companions(self) -> set[tuple]:
        """Sprite kinds that move with the avatar whenever they touch it: its body."""
        out = set()
        for sig, c in self.companion.items():
            n = c["with"] + c["without"]
            if n >= MIN_SUPPORT and c["with"] / n >= PASS_RATE:
                out.add(sig)
        return out

    def body(self, me: Obj | None, objs) -> Obj | None:
        """The avatar together with the companion sprites touching it, as one object
        (the avatar's colour and signature, the union of the pixels)."""
        if me is None:
            return None
        comp = self.companions()
        if not comp:
            return me
        parts = [o for o in objs if o.sig in comp and o is not me and self._touching(me, o)]
        if not parts:
            return me
        pixels = np.concatenate([me.pixels] + [o.pixels for o in parts])
        y1, x1 = pixels.min(axis=0)
        y2, x2 = pixels.max(axis=0)
        return Obj(
            me.color, int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1), len(pixels),
            me.sig, me.esig, pixels,
        )

    def _observe_contact(self, me: Obj, direction, before, after, outcome: str) -> None:
        """Who was touching the avatar's place after the move: on an ordinary step
        the base rate, on a loss the suspects (the frame of a lost game usually
        still shows the scene, with the patrol standing where the avatar went)."""
        comp = self.companions()
        if outcome == OK:
            me2 = self.body(find(after, me.sig), after)
            if me2 is not None:
                for o in after:
                    if o.scenery or o.sig == me.sig or o.sig in comp:
                        continue
                    if self._touching(me2, o):
                        self.near_count[o.sig] += 1
            return
        if outcome != GAME_OVER:
            return
        self.contact_deaths += 1
        dx, dy = direction if direction is not None else (0, 0)
        x1, y1 = min(me.x1, me.x1 + dx), min(me.y1, me.y1 + dy)
        x2, y2 = max(me.x2, me.x2 + dx), max(me.y2, me.y2 + dy)
        was_there = {(o.sig, o.x1, o.y1) for o in before}
        for o in after:
            if o.scenery or o.sig == me.sig or o.sig in comp or self._crosses(o.color):
                continue  # the floor the avatar uncovered by vanishing is not a killer
            if (o.sig, o.x1, o.y1) in was_there:
                continue  # it stood there before the move: it did not come for the avatar
            if o.x1 <= x2 + 1 and o.x2 >= x1 - 1 and o.y1 <= y2 + 1 and o.y2 >= y1 - 1:
                self.killer[o.sig] += 1

    @staticmethod
    def _touching(a: Obj, b: Obj) -> bool:
        return b.x1 <= a.x2 + 1 and b.x2 >= a.x1 - 1 and b.y1 <= a.y2 + 1 and b.y2 >= a.y1 - 1

    def hazard_sigs(self) -> set[tuple]:
        """Object kinds that were touching the avatar's place when the game was
        lost far more often than they are beside it otherwise."""
        if not self.contact_deaths or not self.key_frames:
            return set()
        base = self.contact_deaths / self.key_frames
        out = set()
        for sig, n in self.killer.items():
            total = n + self.near_count[sig]
            if n >= DEATH_SUPPORT and n / total >= GOAL_LIFT * base:
                out.add(sig)
        return out

    def step_size(self) -> int:
        """The longest move the keyboard gives the avatar (1 when unknown)."""
        return max((max(abs(dx), abs(dy)) for dx, dy in self.directions().values()), default=1)

    def hazard_zone(self, objs, shape: tuple[int, int]) -> np.ndarray | None:
        """Cells within one avatar step of an object of a hazard kind, or None."""
        sigs = self.hazard_sigs()
        if not sigs:
            return None
        margin = self.step_size()
        zone = np.zeros(shape, dtype=bool)
        hit = False
        for o in objs:
            if o.sig in sigs:
                hit = True
                zone[max(o.y1 - margin, 0) : o.y2 + margin + 1, max(o.x1 - margin, 0) : o.x2 + margin + 1] = True
        return zone if hit else None

    @staticmethod
    def _swept(
        me: Obj, direction: tuple[int, int], grid: np.ndarray
    ) -> list[tuple[set[int], list[tuple[int, int]]]]:
        """Bands of cells the avatar passes through on the move, nearest first
        (see ``planner.swept``). A move off the grid is one band with the
        pseudo-colour -1 and no cells."""
        rel = me.pixels - np.array([me.y1, me.x1])
        bands = swept(grid, rel, (me.x1, me.y1), direction)
        if bands is None:
            return [({-1}, [])]
        return bands

    def _ahead_cells(
        self, me: Obj, direction: tuple[int, int], grid: np.ndarray
    ) -> tuple[set[int], list[tuple[int, int]]]:
        """Colours and (y, x) cells the avatar passes through on the move, minus its own."""
        colors: set[int] = set()
        coords: list[tuple[int, int]] = []
        for c, cells in self._swept(me, direction, grid):
            colors |= c
            coords.extend(cells)
        return colors, coords

    def _ahead(self, me: Obj, direction: tuple[int, int], grid: np.ndarray) -> set[int]:
        return self._ahead_cells(me, direction, grid)[0]

    def ahead_sigs(self, me: Obj, direction: tuple[int, int], grid: np.ndarray, objs) -> set[tuple]:
        """Signatures of the (non-scenery) objects a move would enter."""
        _, coords = self._ahead_cells(me, direction, grid)
        out = set()
        for y, x in coords:
            o = at(objs, x, y)
            if o is not None and o.sig != me.sig and not o.scenery:
                out.add(o.sig)
        return out

    def _adjacent_sigs(self, me: Obj, objs) -> set[tuple]:
        """Signatures of non-scenery objects whose bounding box touches the avatar's."""
        out = set()
        comp = self.companions()
        for o in objs:
            if o is me or o.scenery or o.sig == me.sig or o.sig in comp:
                continue
            if o.x1 <= me.x2 + 1 and o.x2 >= me.x1 - 1 and o.y1 <= me.y2 + 1 and o.y2 >= me.y1 - 1:
                out.add(o.sig)
        return out

    # -- derived rules --------------------------------------------------------

    def noise(self) -> frozenset:
        """Signatures that change whatever the agent does: particles, timers, animations.

        A signature is noise when it moved, appeared or vanished on at least
        ``NOISE_RATE`` of the transitions under *every* action class that has
        enough observations. Something that reacts to one particular action
        is not noise; something that churns regardless of the action is.
        """
        if self.observations - self._noise_at >= NOISE_REFRESH or self._noise_at == 0:
            self._noise_at = self.observations
            classes = [c for c, n in self.steps_by_class.items() if n >= NOISE_MIN_STEPS]
            out = set(self._noise)  # membership is sticky: a changing set would re-key the graph
            if len(classes) >= 2:
                for sig, c in self.churn.items():
                    if self._controlled(sig):
                        continue
                    if all(c[cls] / self.steps_by_class[cls] >= NOISE_RATE for cls in classes):
                        out.add(sig)
                # Particles clump into rarer shapes of the same colour; take those along.
                colors = {sig[0] for sig in out}
                for sig in self.churn:
                    if sig[0] in colors and sig not in out and not self._controlled(sig):
                        out.add(sig)
            self._noise = frozenset(out)
        return self._noise

    def _controlled(self, sig: tuple) -> bool:
        """Some keyboard action moves this signature in one consistent direction."""
        for (_, s), c in self.move.items():
            if s != sig:
                continue
            nonzero = [(k, n) for k, n in c.items() if k != (0, 0)]
            if not nonzero:
                continue
            best = max(n for _, n in nonzero)
            total = sum(n for _, n in nonzero)
            if best >= MIN_SUPPORT and best / total >= DIR_PURITY:
                return True
        return False

    def avatar(self) -> tuple | None:
        """Signature that the keyboard moves most often."""
        if self._avatar_dirty:
            noise = self.noise()
            moves: Counter = Counter()
            for (_, sig), c in self.move.items():
                if sig in noise:
                    continue
                moves[sig] += sum(n for (dx, dy), n in c.items() if (dx, dy) != (0, 0))
            # The avatar is usually the one object of its kind. A floor made of
            # identical tiles "moves" whenever the avatar covers one of them (the
            # matcher pairs the tile that vanished with the one that reappeared),
            # so moves are weighed against how many instances the kind has.
            def score(sig: tuple) -> float:
                mean_instances = self.instances[sig] / self.key_frames if self.key_frames else 1.0
                return moves[sig] / max(1.0, mean_instances)

            # Ties (a hole inside the avatar moves exactly as often) go to the bigger sprite.
            best = max(moves, key=lambda s: (score(s), s[1] * s[2]), default=None)
            self._avatar = best if best is not None and moves[best] >= MIN_SUPPORT else None
            self._avatar_dirty = False
        return self._avatar

    def direction(self, action_id: int) -> tuple[int, int] | None:
        avatar = self.avatar()
        if avatar is None:
            return None
        c = self.move.get((action_id, avatar))
        if not c:
            return None
        nonzero = [(k, n) for k, n in c.items() if k != (0, 0)]
        if not nonzero:
            return None
        (dx, dy), n = max(nonzero, key=lambda kv: kv[1])
        total = sum(m for _, m in nonzero)
        if n < MIN_SUPPORT or n / total < DIR_PURITY:
            return None
        return dx, dy

    def _rate(self, counter: Counter, key: str) -> tuple[float, int]:
        total = sum(counter.values())
        return (counter[key] / total if total else 0.0), total

    def directions(self) -> dict[int, tuple[int, int]]:
        """Action id -> avatar displacement, for every action with a stable effect."""
        avatar = self.avatar()
        if avatar is None:
            return {}
        out = {}
        for aid in {a for a, s in self.move if s == avatar}:
            d = self.direction(aid)
            if d is not None:
                out[aid] = d
        return out

    def _crosses(self, color: int) -> bool:
        """The avatar has crossed this colour often enough to call it passable."""
        passed, n = self._rate(self.cell[color], "passed")
        return n >= MIN_SUPPORT and passed >= PASS_RATE

    def passable(self, color: int) -> bool:
        """The avatar crossed this colour reliably and never died entering it."""
        visits = self.hazard[color] + sum(self.cell[color].values())
        if visits >= MIN_SUPPORT and self.hazard[color] / visits >= DANGER_RATE:
            return False
        return self._crosses(color)

    def _lifted(self, goals: Counter, visits) -> set:
        """Features under which a level ended far more often than the base rate.

        A feature (a colour ahead, a click target) is a goal when the share of
        its visits that completed a level is at least ``GOAL_LIFT`` times the
        share of all transitions that did. One event on a rare feature counts;
        one event on something seen thousands of times does not.
        """
        if not self.level_ups or not self.observations:
            return set()
        base = self.level_ups / self.observations
        out = set()
        for f, n in goals.items():
            total = visits(f)
            if total and n / total >= GOAL_LIFT * base:
                out.add(f)
        return out

    def goal_colors(self) -> set[int]:
        """Colours ahead of the avatar when a level ended, beyond chance."""
        return self._lifted(
            self.goal_ahead,
            lambda c: self.goal_ahead[c] + sum(self.cell[c].values()) + self.hazard[c],
        )

    def goal_sigs(self) -> set[tuple]:
        """Object signatures the avatar entered when a level ended, beyond chance.

        Finer than ``goal_colors``: in ls20 the exit and the walls share a colour,
        so a colour goal sends the planner into walls; the exit's signature does not.
        """
        return self._lifted(self.goal_ahead_sig, lambda sig: self.touch_count[sig])

    def goal_adjacent_sigs(self) -> set[tuple]:
        """Signatures that touched the avatar when a level ended, beyond chance.

        For games where the avatar pushes or carries things (tu93's blocks), the
        object that matters is beside the avatar, not in the cell it enters.
        """
        return self._lifted(self.goal_adjacent, lambda sig: self.adjacent_count[sig])

    def goal_key_actions(self) -> set[int]:
        """Keyboard actions that completed levels beyond chance (an 'interact' key).

        Keys are pressed constantly, so the lift test alone rarely fires; a key
        that completed at least two levels and nearly all of them also counts.
        """
        out = self._lifted(self.goal_action, lambda aid: self.key_uses[aid])
        for aid, n in self.goal_action.items():
            if n >= 2 and n / self.level_ups >= 0.8:
                out.add(aid)
        return out

    def goal_click_sigs(self) -> set[tuple]:
        return self._lifted(self.goal_click, lambda sig: sum(self.click[sig].values()))

    def goal_click_colors(self) -> set[int]:
        return self._lifted(self.goal_click_color, lambda c: self.click_color[c])

    # -- prediction -----------------------------------------------------------

    def predict(
        self, action: GameAction, data: dict[str, int] | None, objs: list[Obj], grid: np.ndarray
    ) -> Prediction:
        if action.is_complex():
            if data is None:
                return Prediction("unknown")
            target = at(objs, data["x"], data["y"])
            if target is None:
                return Prediction("unknown")
            inst = (target.esig, target.x1, target.y1)
            # Danger generalises per instance (this cell kills) and per kind; a no-op
            # rule only per kind: a cell that did nothing so far may matter later, and
            # skipping it per instance cost levels on r11l and su15.
            for deaths, counter, where in (
                (self.click_death_at[inst], self.click_at[inst], f"{target.label()} at {target.x1},{target.y1}"),
                (self.click_death[target.esig], self.click[target.esig], target.label()),
            ):
                n = sum(counter.values())
                if deaths >= DEATH_SUPPORT and deaths / n >= DANGER_RATE:
                    return Prediction("danger", n, f"clicking {where} lost the game {deaths}/{n}x")
            rate, n = self._rate(self.click[target.esig], "none")
            if n >= MIN_SUPPORT and rate >= NOOP_RATE:
                return Prediction("noop_click", n, f"clicking {target.label()} changed nothing {n}x")
            return Prediction("unknown", n)

        # An action that loses the game whatever the situation (sp80's ACTION5 did so
        # 145 times in one run) is dangerous regardless of what is ahead.
        deaths, uses = self.key_death[action.value], self.key_uses[action.value]
        if deaths >= MIN_SUPPORT and deaths / uses >= DANGER_RATE:
            return Prediction("danger", uses, f"{action.name} lost the game {deaths}/{uses}x")
        avatar = self.avatar()
        direction = self.direction(action.value)
        if avatar is None or direction is None:
            return Prediction("unknown")
        me = self.body(find(objs, avatar), objs)
        if me is None:
            return Prediction("unknown")
        zone = self.hazard_zone(objs, grid.shape)
        if zone is not None:
            dx, dy = direction
            ys, xs = me.pixels[:, 0], me.pixels[:, 1]
            here = zone[ys, xs].any()
            ys2, xs2 = np.clip(ys + dy, 0, grid.shape[0] - 1), np.clip(xs + dx, 0, grid.shape[1] - 1)
            if zone[ys2, xs2].any() and not here:
                return Prediction("danger", self.contact_deaths, "a kind that kills on contact is within a step")
        colors = self._ahead(me, direction, grid)
        verdict_danger = verdict_block = None
        all_pass = True
        support = 0
        for c in colors:
            visits = self.hazard[c] + sum(self.cell[c].values())
            if visits >= MIN_SUPPORT and self.hazard[c] / visits >= DANGER_RATE:
                verdict_danger = (c, visits)
            blocked, n = self._rate(self.cell[c], "blocked")
            if n >= MIN_SUPPORT and blocked >= BLOCK_RATE:
                verdict_block = (c, n)
            passed, n = self._rate(self.cell[c], "passed")
            if not (n >= MIN_SUPPORT and passed >= PASS_RATE):
                all_pass = False
            support += n
        if verdict_danger:
            c, n = verdict_danger
            return Prediction("danger", n, f"colour {c} ahead ended the game before")
        if verdict_block:
            c, n = verdict_block
            return Prediction("blocked", n, f"colour {c} ahead blocks the avatar")
        if all_pass and colors:
            return Prediction("move", support, f"avatar moves {direction}")
        return Prediction("unknown", support)

    # -- reporting ------------------------------------------------------------

    def describe(self) -> list[str]:
        lines: list[str] = [f"{self.observations} transitions observed"]
        for sig in sorted(self.noise(), key=repr):
            total = sum(self.churn[sig].values())
            lines.append(
                f"noise: colour-{sig[0]} {sig[2]}x{sig[1]} changes whatever the action ({total} events)"
            )
        for aid, n in self.key_death.most_common():
            if n >= MIN_SUPPORT:
                lines.append(f"ACTION{aid} lost the game {n}/{self.key_uses[aid]} times")
        avatar = self.avatar()
        if avatar is None:
            lines.append("no avatar found (nothing moves consistently under the keyboard)")
        else:
            lines.append(f"avatar: colour-{avatar[0]} {avatar[2]}x{avatar[1]} sprite")
            for sig in sorted(self.companions(), key=repr):
                c = self.companion[sig]
                lines.append(f"  moves with it as one body: colour-{sig[0]} {sig[2]}x{sig[1]} ({c['with']}/{c['with'] + c['without']})")
            for aid in sorted({a for a, s in self.move if s == avatar}):
                c = self.move[(aid, avatar)]
                d = self.direction(aid)
                total = sum(c.values())
                if d is None:
                    lines.append(f"  ACTION{aid}: no consistent effect ({dict(c)})")
                else:
                    stuck = c[(0, 0)]
                    lines.append(
                        f"  ACTION{aid}: moves it by {d} ({total - stuck}/{total} times; "
                        f"stuck {stuck}x)"
                    )
            for color, c in sorted(self.cell.items()):
                blocked, n = self._rate(c, "blocked")
                if n >= MIN_SUPPORT:
                    what = "edge" if color == -1 else f"colour {color}"
                    if blocked >= BLOCK_RATE:
                        lines.append(f"  {what} blocks the avatar ({c['blocked']}/{n})")
                    elif 1 - blocked >= PASS_RATE:
                        lines.append(f"  {what} is walkable ({c['passed']}/{n})")
                    else:
                        lines.append(
                            f"  {what} ahead: blocked {c['blocked']}/{n} times "
                            "(depends on something else)"
                        )
            budget = self.budget()
            if budget is not None:
                lines.append(f"every attempt ends after {budget} actions (a move budget)")
            for sig in sorted(self.hazard_sigs(), key=repr):
                lines.append(
                    f"  colour-{sig[0]} {sig[2]}x{sig[1]} kills on contact ({self.killer[sig]}/{self.killer[sig] + self.near_count[sig]} contacts)"
                )
            for t in sorted(self.switches(), key=repr):
                n = self.tile_entered[t]
                effects = [f"{what} colour-{sig[0]} {sig[2]}x{sig[1]} ({k}/{n})" for (what, sig), k in self.remote[t].items() if k / n >= REMOTE_RATE]
                lines.append(f"  entering colour-{t[0]} {t[2]}x{t[1]} is a switch: far away, {'; '.join(effects)}")
            for color, n in self.hazard.most_common():
                visits = n + sum(self.cell[color].values())
                lines.append(f"  colour {color} ahead ended the game {n}/{visits} times")
            for color in sorted(self.goal_colors()):
                lines.append(f"  entering colour {color} completed a level ({self.goal_ahead[color]}x)")
            for sig in sorted(self.goal_sigs(), key=repr):
                lines.append(
                    f"  entering the colour-{sig[0]} {sig[2]}x{sig[1]} sprite completed a level "
                    f"({self.goal_ahead_sig[sig]}/{self.touch_count[sig]} contacts)"
                )
            if self.goal_adjacent:
                near = ", ".join(
                    f"colour-{s[0]} {s[2]}x{s[1]} ({n}/{self.adjacent_count[s]})"
                    for s, n in self.goal_adjacent.most_common(4)
                )
                lines.append(f"  objects touching the avatar when levels ended: {near}")
            for sig in sorted(self.goal_adjacent_sigs(), key=repr):
                lines.append(
                    f"  being next to the colour-{sig[0]} {sig[2]}x{sig[1]} sprite is a goal condition"
                )
        for sig in self.goal_click_sigs():
            lines.append(
                f"clicking colour-{sig[0]} {sig[2]}x{sig[1]} completed a level ({self.goal_click[sig]}x)"
            )
        for c in sorted(self.goal_click_colors()):
            lines.append(
                f"clicking colour {c} completed a level ({self.goal_click_color[c]}/{self.click_color[c]} clicks)"
            )
        for sig, n in self.click_death.most_common(6):
            total = sum(self.click[sig].values())
            lines.append(f"clicking colour-{sig[0]} {sig[2]}x{sig[1]} lost the game {n}/{total} times")
        other = [
            (aid, sig, c)
            for (aid, sig), c in self.move.items()
            if sig != avatar and sum(n for k, n in c.items() if k != (0, 0)) >= MIN_SUPPORT
        ]
        for aid, sig, c in sorted(other, key=lambda t: -sum(t[2].values()))[:8]:
            lines.append(f"ACTION{aid} also moves colour-{sig[0]} {sig[2]}x{sig[1]}: {dict(c)}")
        noops = [(sig, c) for sig, c in self.click.items() if sum(c.values()) >= MIN_SUPPORT]
        for sig, c in sorted(noops, key=lambda t: -sum(t[1].values()))[:12]:
            total = sum(c.values())
            lines.append(
                f"click on colour-{sig[0]} {sig[2]}x{sig[1]}: changed something "
                f"{c['change']}/{total} times"
            )
        return lines
