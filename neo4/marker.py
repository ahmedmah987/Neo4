"""The marker family: bring pieces onto same-coloured markers.

Many puzzles show a piece of some colour and, elsewhere, a small static mark
of the same colour (a ring, four corner dots, a cross, a dot), and are solved
when the piece covers the mark. The piece is moved indirectly: by buttons that
rotate a ring of tiles, by sliders that extend an arm, by keys that move a
selected shape. This solver learns, from observed transitions, what every
action does to every moving piece, and plans in that learned model: a
breadth-first search over piece positions for the shortest action sequence
after which every mark is covered by a piece of its colour.

Perception (no game-specific knowledge):

* pieces are single-colour connected components, matched between frames by
  colour and exact shape with a minimum-squared-distance assignment, so a ring
  of identical tiles that shifts by one cell reads as a uniform shift;
* a *mover* is a component that has been displaced at least once this level;
* *marks* are clusters of small static same-colour components (four corner
  dots form one mark) taken from the level's first frame, whose colour is also
  the colour of a mover and whose shape is not a mover's shape.

Learned model of an action (an *operator*):

* a position map: a component of shape S at position p went to p' (exact for
  position-bound mechanics such as rotating rings, and it records blocked
  moves as p -> p);
* a translation: the components it moved last time move by d again (for a
  selected piece, an arm tip); when an action changes the frame without moving
  anything (a selection change), every translation goes stale and is re-bound
  to whatever moves next by the same displacement.

The solver only observes while the fallback agent explores; it takes control
once its family is recognised (a mover and a mark of the same colour exist)
and a plan or a useful probe is available, and hands control back when the
model keeps mispredicting or the marks are covered without the level ending.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp
from .objects import _exact_hash

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - only where scipy is absent
    linear_sum_assignment = None

Action = tuple[GameAction, dict[str, int] | None]
OpKey = tuple  # ("k", action id) or ("c", x, y)

MAX_AREA = 400  # larger components are scenery
MAX_MOVE = 96  # no practical limit: a ring of tiles wraps around, a piece can teleport
MARK_GAP = 4  # small static components this close together form one mark
MARK_SIZE = 12  # a mark fits in this many pixels square
MAX_PROBES = 30  # probes per level
MAX_MISMATCHES = 20  # mispredicted actions per level before giving the level back
MAX_EXPANSIONS = 60000
LONG_PLAN = 3  # before running a longer plan, probe look-alike controls not yet tried


@dataclass(slots=True)
class Comp:
    color: int
    shape: str  # exact shape hash
    kind: str  # colour-free, orientation-free shape hash: what a control "looks like"
    x: int
    y: int
    w: int
    h: int
    area: int
    rel: frozenset  # (dy, dx) pixel offsets from the top-left
    point: tuple[int, int]  # a click point on the component, (x, y)
    id: int = -1

    @property
    def key(self) -> tuple:
        return (self.color, self.shape, self.x, self.y)


def kind_of(c: fp.Component) -> str:
    """Hash of a component's shape that ignores colour, rotation, mirroring and
    pixel scale: a control drawn at twice the size (a level shown at scale 2)
    is the same kind of control."""
    local = np.zeros((c.height, c.width), dtype=bool)
    local[c.pixels[:, 0] - c.y1, c.pixels[:, 1] - c.x1] = True
    for k in (4, 3, 2):
        h, w = local.shape
        if h % k == 0 and w % k == 0:
            small = local[::k, ::k]
            if np.array_equal(np.kron(small, np.ones((k, k), dtype=bool)), local):
                local = small
                break
    variants = []
    for r in range(4):
        m = np.rot90(local, r)
        variants.append(m)
        variants.append(np.fliplr(m))
    canon = min(variants, key=lambda m: (m.shape, np.packbits(m).tobytes()))
    return hashlib.blake2b(np.packbits(canon).tobytes() + bytes(canon.shape), digest_size=6).hexdigest()


def components(grid: np.ndarray, mask: np.ndarray | None) -> list[Comp]:
    out = []
    for c in fp.segment(fp.apply_mask(grid, mask)):
        if c.color == fp.MASK_COLOR or c.area > MAX_AREA:
            continue
        rel = frozenset(zip((c.pixels[:, 0] - c.y1).tolist(), (c.pixels[:, 1] - c.x1).tolist()))
        out.append(Comp(c.color, _exact_hash(c), kind_of(c), c.x1, c.y1, c.width, c.height,
                        c.area, rel, c.click_point()))
    return out


def _step_cost(dx: int, dy: int) -> float:
    """Cost of explaining a component's change of place by a move of (dx, dy):
    path length, so one long jump (a ring wrapping round) is not traded for a
    reshuffle of many neighbours, plus a small quadratic term that breaks ties
    in favour of uniform shifts (a row of identical tiles moving one cell
    rather than one tile jumping two)."""
    return float(np.hypot(dx, dy)) + 0.01 * (dx * dx + dy * dy)


def match(before: list[Comp], after: list[Comp]) -> dict[int, tuple[int, int]]:
    """Carry ids from ``before`` to ``after``; return {id: displacement} of the
    components that moved. Components of one colour and shape are paired by a
    minimum total squared distance, which reads a shifting row of identical
    tiles as one uniform shift."""
    groups: dict[tuple, tuple[list[Comp], list[Comp]]] = {}
    for c in before:
        groups.setdefault((c.color, c.shape), ([], []))[0].append(c)
    for c in after:
        groups.setdefault((c.color, c.shape), ([], []))[1].append(c)
    moved: dict[int, tuple[int, int]] = {}
    for b, a in groups.values():
        if not b or not a:
            continue
        cost = np.array([[_step_cost(q.x - p.x, q.y - p.y) for q in a] for p in b], dtype=float)
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            pairs = list(zip(rows.tolist(), cols.tolist()))
        else:  # greedy fallback
            pairs, used_b, used_a = [], set(), set()
            for i, j in sorted(np.ndindex(cost.shape), key=lambda ij: cost[ij]):
                if i not in used_b and j not in used_a:
                    pairs.append((i, j)); used_b.add(i); used_a.add(j)
        for i, j in pairs:
            if cost[i, j] > _step_cost(MAX_MOVE, MAX_MOVE):
                continue
            p, q = b[i], a[j]
            q.id = p.id
            if (p.x, p.y) != (q.x, q.y):
                moved[p.id] = (q.x - p.x, q.y - p.y)
    return moved


@dataclass
class Op:
    """What one action has been seen to do.

    ``cands`` treats the action as a fixed map of places (a ring of tiles
    turning, a conveyor): for a component of shape S at p, the places q that
    are consistent with every observation so far, i.e. after the action q
    showed the colour p showed before. One observation of a ring of distinct
    colours pins the map down; identical tiles leave ambiguity that later
    observations remove. Until then the nearest candidate is the guess.

    ``movers``/``d`` treat it as a translation of the components it moved last
    time (a selected piece, an arm tip), which carries on beyond the places
    seen, but only while every component it moved moved the same way.
    """

    cands: dict[tuple, set] = field(default_factory=dict)  # (shape, x, y) -> {(x, y)}
    best: dict[tuple, tuple[int, int]] = field(default_factory=dict)  # the current guess
    dirty: set[str] = field(default_factory=set)  # shapes whose guess must be recomputed
    movers: set[int] = field(default_factory=set)  # ids it moved last time
    d: tuple[int, int] | None = None  # the displacement it gave most of them
    stale: bool = False  # a selection change since: re-bind on the next observed move
    uniform: bool = True  # every component it moved moved the same way (a translation)
    tried: int = 0

    def learn(self, before: list[Comp], after: list[Comp]) -> None:
        """Narrow the candidate destinations with one observed transition, for every
        shape whose components changed place."""
        old: dict[str, Counter] = {}
        new: dict[str, Counter] = {}
        for c in before:
            old.setdefault(c.shape, Counter())[(c.x, c.y, c.color)] += 1
        for c in after:
            new.setdefault(c.shape, Counter())[(c.x, c.y, c.color)] += 1
        for shape, b in old.items():
            a = new.get(shape, Counter())
            if a == b:
                continue
            by_color: dict[int, set] = {}
            for (x, y, col) in a:
                by_color.setdefault(col, set()).add((x, y))
            for (x, y, col) in b:
                q = by_color.get(col, set())
                key = (shape, x, y)
                prev = self.cands.get(key)
                both = prev & q if prev is not None else None
                self.cands[key] = set(both) if both else set(q)
            self.dirty.add(shape)

    def _assign(self, shape: str) -> None:
        """The guess for every place of one shape: a one-to-one map into the
        candidate places with the least total path length, so a ring guessed
        from identical tiles still closes into a cycle."""
        keys = [k for k in self.cands if k[0] == shape]
        dests = sorted({q for k in keys for q in self.cands[k]})
        if not keys or not dests:
            return
        col = {q: j for j, q in enumerate(dests)}
        big = 1e6
        cost = np.full((len(keys), len(dests)), big)
        for i, k in enumerate(keys):
            for q in self.cands[k]:
                cost[i, col[q]] = _step_cost(q[0] - k[1], q[1] - k[2])
        if linear_sum_assignment is None:
            return
        for k in keys:
            self.best.pop(k, None)
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if cost[i, j] < big:
                self.best[keys[i]] = dests[j]

    def block(self, shape: str, pos: tuple[int, int]) -> None:
        self.cands[(shape, pos[0], pos[1])] = {pos}
        self.best[(shape, pos[0], pos[1])] = pos

    def predict(self, c_shape: str, cid: int, pos: tuple[int, int]) -> tuple[int, int]:
        if c_shape in self.dirty:
            self.dirty.discard(c_shape)
            self._assign(c_shape)
        b = self.best.get((c_shape, pos[0], pos[1]))
        if b is not None:
            return b
        c = self.cands.get((c_shape, pos[0], pos[1]))
        if c:
            return min(c, key=lambda q: ((q[0] - pos[0]) ** 2 + (q[1] - pos[1]) ** 2, q))
        # A translation (a selected piece, an arm tip) carries on beyond the
        # positions seen; a permutation (a ring of tiles) leaves unseen places alone.
        if self.uniform and not self.stale and self.d is not None and cid in self.movers:
            return (pos[0] + self.d[0], pos[1] + self.d[1])
        return pos

    def options(self, c_shape: str, cid: int, pos: tuple[int, int], n: int = 3) -> list[tuple[int, int]]:
        """The guess and the next nearest candidates: what the action may do here."""
        first = self.predict(c_shape, cid, pos)
        c = self.cands.get((c_shape, pos[0], pos[1]))
        if not c:
            return [first]
        near = sorted(c, key=lambda q: ((q[0] - pos[0]) ** 2 + (q[1] - pos[1]) ** 2, q))
        return [first] + [q for q in near if q != first][: n - 1]

    @property
    def effective(self) -> bool:
        return self.d is not None or any(len(v) != 1 or next(iter(v)) != k[1:] for k, v in self.cands.items())


def mark_clusters(comps: list[Comp]) -> list[list[Comp]]:
    """Small same-colour components close together, as clusters (union-find)."""
    small = [c for c in comps if c.w <= MARK_SIZE and c.h <= MARK_SIZE]
    parent = list(range(len(small)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(small):
        for j in range(i + 1, len(small)):
            b = small[j]
            if a.color != b.color:
                continue
            gx = max(b.x - (a.x + a.w - 1), a.x - (b.x + b.w - 1), 0) - 1
            gy = max(b.y - (a.y + a.h - 1), a.y - (b.y + b.h - 1), 0) - 1
            if max(gx, gy) <= MARK_GAP:
                parent[root(i)] = root(j)
    out: dict[int, list[Comp]] = {}
    for i, c in enumerate(small):
        out.setdefault(root(i), []).append(c)
    clusters = []
    for cl in out.values():
        x1 = min(c.x for c in cl); x2 = max(c.x + c.w - 1 for c in cl)
        y1 = min(c.y for c in cl); y2 = max(c.y + c.h - 1 for c in cl)
        if x2 - x1 < MARK_SIZE and y2 - y1 < MARK_SIZE:
            clusters.append(cl)
    return clusters


def centre(cl: list[Comp]) -> tuple[float, float]:
    x1 = min(c.x for c in cl); x2 = max(c.x + c.w - 1 for c in cl)
    y1 = min(c.y for c in cl); y2 = max(c.y + c.h - 1 for c in cl)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def covers(rel: frozenset, pos: tuple[int, int], pt: tuple[float, float]) -> bool:
    """Does a component with pixel offsets ``rel`` at ``pos`` have a pixel
    within half a pixel of the point ``pt`` (x, y)?"""
    cx, cy = pt
    for x in {int(np.floor(cx)), int(np.ceil(cx))}:
        for y in {int(np.floor(cy)), int(np.ceil(cy))}:
            if abs(x - cx) <= 0.5 and abs(y - cy) <= 0.5 and (y - pos[1], x - pos[0]) in rel:
                return True
    return False


class MarkerSolver:
    name = "marker"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.confirmed = False  # it completed a level of this game itself
        self.disabled = False
        self.strikes = 0
        self.control_kinds: Counter = Counter()  # click-target kinds that moved a mover
        self.key_ops: Counter = Counter()  # keyboard actions that moved a mover
        self.prior_colors: set[int] = set()  # colours of the marks of earlier levels
        self.in_control = False
        self.stats: Counter = Counter()
        self._next_id = 0
        self.level = -1
        self._new_level(None)

    # -- bookkeeping ------------------------------------------------------------

    def _new_level(self, grid: np.ndarray | None) -> None:
        self.mask = fp.status_bar_mask(fp.segment(grid)) if grid is not None else None
        self.first: list[Comp] = []
        self.comps: list[Comp] = []
        self.start_ids: dict[tuple, int] = {}
        self.ops: dict[OpKey, Op] = {}
        self.movers: set[int] = set()
        self.mover_shapes: set[tuple] = set()  # (colour, shape) of movers
        self.probed: set[OpKey] = set()
        self.probes = 0
        self.mismatches = 0
        self.plan: list[OpKey] = []
        self.expect: dict[int, tuple[int, int]] | None = None
        self.gave_up = False
        self.deaths = 0
        self.mark_ids: set[int] = set()
        if grid is not None:
            self.comps = components(grid, self.mask)
            for c in self.comps:
                c.id = self._new_id()
            self.first = list(self.comps)
            self.start_ids = {c.key: c.id for c in self.comps}

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def op_key(self, action: GameAction, data: dict | None) -> OpKey:
        if action.is_complex() and data is not None:
            return ("c", int(data["x"]), int(data["y"]))
        return ("k", action.value)

    # -- observing --------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        """Update the model with the transition that led to ``obs``."""
        if obs.state == GameState.GAME_OVER and self.in_control:
            self.deaths += 1
            self.stats["deaths"] += 1
        if obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        grid = np.asarray(obs.frame[-1])
        if obs.levels_completed != self.level:
            if self.in_control and obs.levels_completed > self.level >= 0:
                self.confirmed = True
                self.stats["levels"] += 1
                self._remember_controls()
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level(grid)
            return
        after = components(grid, self.mask)
        if action is None or action is GameAction.RESET:
            # a fresh attempt at the same level: the components are the level's start
            for c in after:
                c.id = self.start_ids.get(c.key, self._new_id())
            self.comps = after
            self.plan = []
            self.expect = None
            return
        moved = match(self.comps, after)
        for c in after:
            if c.id < 0:
                c.id = self._new_id()
        key = self.op_key(action, data)
        op = self.ops.setdefault(key, Op())
        op.tried += 1
        before = {c.id: c for c in self.comps}
        op.learn(self.comps, after)
        if moved:
            for cid, d in moved.items():
                b = before[cid]
                self.movers.add(cid)
                self.mover_shapes.add((b.color, b.shape))
            common = Counter(moved.values()).most_common(1)[0][0]
            ids = {cid for cid, d in moved.items() if d == common}
            for other in self.ops.values():
                if other.stale and other.d is not None and other is not op:
                    other.movers = set(ids)  # the selection moved: re-bind
                    other.stale = False
            op.movers, op.d, op.stale = set(ids), common, False
            op.uniform = op.uniform and len(set(moved.values())) == 1
        else:
            changed = {c.key for c in self.comps} != {c.key for c in after}
            if changed:  # something changed but nothing moved: a selection change
                for other in self.ops.values():
                    if other is not op:
                        other.stale = True
        # A relevant mover the model expected to move but which stayed: blocked here.
        if self.expect is not None:
            now = {c.id: (c.x, c.y) for c in after}
            for cid, pos in self.expect.items():
                b = before.get(cid)
                if b is None:
                    continue
                if now.get(cid) != pos:
                    self.mismatches += 1
                    self.stats["mismatches"] += 1
                    if now.get(cid) == (b.x, b.y):
                        op.block(b.shape, (b.x, b.y))
                    self.plan = []
                    break
            self.expect = None
        self.comps = after

    def _remember_controls(self) -> None:
        for key, op in self.ops.items():
            if not (op.movers & self.movers) or op.d is None:
                continue
            if key[0] == "k":
                self.key_ops[key[1]] += 1
            else:
                c = self._comp_at(self.first, key[1], key[2]) or self._comp_at(self.comps, key[1], key[2])
                if c is not None:
                    self.control_kinds[c.kind] += 1
        self.prior_colors = {col for col, _ in self._groups()[0]}

    def _effective_kinds(self) -> set[str]:
        """Kinds of the click targets whose clicks moved a mover, this level and before."""
        kinds = set(self.control_kinds)
        for key, op in self.ops.items():
            if key[0] == "c" and op.d is not None and op.movers:
                c = self._comp_at(self.first, key[1], key[2])
                if c is not None:
                    kinds.add(c.kind)
        return kinds

    @staticmethod
    def _comp_at(comps: list[Comp], x: int, y: int) -> Comp | None:
        for c in comps:
            if (y - c.y, x - c.x) in c.rel:
                return c
        return None

    # -- the goal ---------------------------------------------------------------

    def _groups(self) -> tuple[list[tuple[int, list[tuple[float, float]]]], list[Comp]]:
        """(colour, mark points) for every colour that has movers and marks, and the
        movers of those colours (current components)."""
        mover_colors = {col for col, _ in self.mover_shapes}
        marks: dict[int, list[tuple[float, float]]] = {}
        static = [c for c in self.first if c.id not in self.movers and c.color in mover_colors]
        for cl in mark_clusters(static):
            if len(cl) == 1 and (cl[0].color, cl[0].shape) in self.mover_shapes:
                continue  # looks like a piece that has not moved yet
            marks.setdefault(cl[0].color, []).append(centre(cl))
        groups = [(col, pts) for col, pts in sorted(marks.items())]
        self.mark_ids = {c.id for cl in mark_clusters(static) for c in cl if c.color in marks}
        colors = {col for col, _ in groups}
        movers = [c for c in self.comps if c.color in colors and (c.color, c.shape) in self.mover_shapes]
        return groups, movers

    def recognised(self) -> bool:
        return bool(self._groups()[0])

    # -- planning ---------------------------------------------------------------

    def _satisfied(self, groups, movers, state) -> list[bool]:
        out = []
        for col, pts in groups:
            idx = [i for i, m in enumerate(movers) if m.color == col]
            out.append(all(any(covers(movers[i].rel, state[i], p) for i in idx) for p in pts))
        return out

    def _search(self, groups, movers, need, optimistic: bool = False) -> list[OpKey] | None:
        """Shortest op sequence after which the groups flagged in ``need`` are
        satisfied. ``optimistic``: where the model is still unsure what an action
        does at a place, any of the nearest candidates may happen; running such a
        plan is also how the uncertainty gets resolved."""
        ops = [(k, op) for k, op in self.ops.items() if op.d is not None or op.cands]
        start = tuple((m.x, m.y) for m in movers)

        def done(state) -> bool:
            sat = self._satisfied(groups, movers, state)
            return all(s for s, n in zip(sat, need) if n)

        def successors(state, op):
            if not optimistic:
                yield tuple(op.predict(m.shape, m.id, pos) for m, pos in zip(movers, state))
                return
            choices = [op.options(m.shape, m.id, pos) for m, pos in zip(movers, state)]
            combos = [()]
            for ch in choices:
                combos = [c + (q,) for c in combos for q in ch][:27]
            yield from combos

        if done(start):
            return []
        parent: dict[tuple, tuple[tuple, OpKey]] = {}
        seen = {start}
        queue = deque([start])
        expansions = 0
        while queue and expansions < MAX_EXPANSIONS:
            state = queue.popleft()
            expansions += 1
            for key, op in ops:
                for nxt in successors(state, op):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    parent[nxt] = (state, key)
                    if done(nxt):
                        path = []
                        while nxt != start:
                            nxt, k = parent[nxt]
                            path.append(k)
                        return path[::-1]
                    queue.append(nxt)
        return None

    def _make_plan(self) -> list[OpKey] | None:
        groups, movers = self._groups()
        if not groups or not movers:
            return None
        plan = self._search(groups, movers, [True] * len(groups))
        if plan is not None:
            return plan
        plan = self._search(groups, movers, [True] * len(groups), optimistic=True)
        if plan is not None:
            self.stats["optimistic_plans"] += 1
            return plan
        # A decoy colour can make the joint goal impossible: satisfy groups one at a
        # time, shortest first, keeping those already satisfied.
        sat = self._satisfied(groups, movers, tuple((m.x, m.y) for m in movers))
        best = None
        for i, s in enumerate(sat):
            if s:
                continue
            need = [j == i or sat[j] for j in range(len(groups))]
            p = self._search(groups, movers, need)
            if p is None:
                p = self._search(groups, movers, [j == i for j in range(len(groups))])
            if p is not None and (best is None or len(p) < len(best)):
                best = p
        return best

    # -- probing ----------------------------------------------------------------

    def _probe(self, obs: FrameDataRaw, lookalike: bool = False) -> OpKey | None:
        """The next action to try. ``lookalike``: only click targets shaped like a
        control that has moved something (controls tend to come in families,
        such as the two arrows of a ring)."""
        if self.probes >= MAX_PROBES:
            return None
        kinds = self._effective_kinds() if lookalike else set()
        if lookalike and not kinds:
            return None
        keys = [] if lookalike else [a for a in obs.available_actions if a not in (0, 6)]
        # keyboard actions first: known movers first, undo (7) last
        keys.sort(key=lambda a: (-self.key_ops[a], a == 7, a))
        for a in keys:
            k = ("k", a)
            op = self.ops.get(k)
            if k not in self.probed and (op is None or op.stale):
                return k
        if 6 not in obs.available_actions:
            return None
        groups, movers = self._groups()
        anchors = [(m.x + m.w / 2, m.y + m.h / 2) for m in movers]
        if not anchors:
            anchors = [(c.x + c.w / 2, c.y + c.h / 2) for c in self.comps if c.color in self.prior_colors]
        used_keys = set()
        for k in self.probed:
            if k[0] == "c":
                hit = self._comp_at(self.comps, k[1], k[2])
                if hit is not None:
                    used_keys.add(hit.key)
        known_points = {(k[1], k[2]) for k in self.ops if k[0] == "c"}
        cands = []
        for c in self.comps:
            if c.id in self.movers or c.id in self.mark_ids or c.key in used_keys or c.point in known_points:
                continue
            if lookalike and c.kind not in kinds:
                continue
            dist = min((abs(c.x + c.w / 2 - ax) + abs(c.y + c.h / 2 - ay) for ax, ay in anchors), default=0.0)
            cands.append((-self.control_kinds[c.kind], dist, c.y, c.x, c))
        if not cands:
            return None
        cands.sort(key=lambda t: t[:4])
        c = cands[0][-1]
        return ("c", c.point[0], c.point[1])

    # -- acting -----------------------------------------------------------------

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.strikes += 1
        self.stats["give_ups"] += 1
        if self.strikes >= (2 if self.confirmed else 1):
            self.disabled = True

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not self.in_control:
            if not (self.recognised() or (self.confirmed and self.probes == 0 and not self.ops)):
                return None
            self.in_control = True
            self.stats["takeovers"] += 1
        if self.mismatches > MAX_MISMATCHES:
            self._give_up()
            return None
        if not self.plan:
            plan = self._make_plan() if self.recognised() else None
            if plan == []:
                self._give_up()  # every mark is covered and the level has not ended
                return None
            if plan is not None and len(plan) > LONG_PLAN and (self.deaths or self.probes < len(plan) / 2):
                key = self._probe(obs, lookalike=True)
                if key is not None:
                    self.probed.add(key)
                    self.probes += 1
                    self.stats["lookalike_probes"] += 1
                    return self._action(key)
            if plan is not None:
                self.plan = plan
                self.stats["plans"] += 1
        if self.plan:
            key = self.plan.pop(0)
            groups, movers = self._groups()
            op = self.ops[key]
            self.expect = {m.id: op.predict(m.shape, m.id, (m.x, m.y)) for m in movers}
            self.stats["plan_steps"] += 1
            return self._action(key)
        key = self._probe(obs)
        if key is None:
            self._give_up()
            return None
        self.probed.add(key)
        self.probes += 1
        self.stats["probes"] += 1
        return self._action(key)

    @staticmethod
    def _action(key: OpKey) -> Action:
        if key[0] == "k":
            return GameAction.from_id(key[1]), None
        return GameAction.ACTION6, {"x": key[1], "y": key[2]}
