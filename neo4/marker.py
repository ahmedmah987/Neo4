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
MAX_PROBES = 60  # probes per level
UNPROVEN_PROBES = 15  # probes per level in a game the solver has not solved a level of
MAX_MISMATCHES = 20  # mispredicted actions per level before giving the level back
MAX_EXPANSIONS = 150000  # predictions (actions x pieces) per search
PLAN_BUDGET = 300000  # predictions over all the searches of one planning call
LEVEL_WORK = 3000000  # predictions a level may spend on searches while another agent plays
LONG_PLAN = 3  # before running a longer plan, probe look-alike controls not yet tried
MAX_OPTIONS = 8  # candidate places beyond which an uncertain action is not enumerated
SCROLL_SHARE = 0.6  # a transition moving this share of the screen's components alike is a camera scroll
PLAN_EVERY = 5  # observations between planning attempts while another agent plays
SCENERY_COPIES = 12  # a component with this many identical copies on screen is scenery (a floor of cells)
MAX_DEATHS = 3  # lost attempts under the solver per level before giving the level back
MAX_LEVEL_ACTIONS = 400  # actions the solver may spend on one level


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


def exact_shape(c: fp.Component) -> str:
    """Hash of a component's exact pixel mask, dimensions included (a 1x11 and
    an 11x1 line have the same bits but are different shapes)."""
    local = np.zeros((c.height, c.width), dtype=bool)
    local[c.pixels[:, 0] - c.y1, c.pixels[:, 1] - c.x1] = True
    return hashlib.blake2b(np.packbits(local).tobytes() + bytes(local.shape), digest_size=6).hexdigest()


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


_KINDS: dict[str, str] = {}  # exact shape -> kind: the kind depends on the mask only


def components(grid: np.ndarray, mask: np.ndarray | None) -> list[Comp]:
    out = []
    for c in fp.segment(fp.apply_mask(grid, mask)):
        if c.color == fp.MASK_COLOR or c.area > MAX_AREA:
            continue
        rel = frozenset(zip((c.pixels[:, 0] - c.y1).tolist(), (c.pixels[:, 1] - c.x1).tolist()))
        shape = exact_shape(c)
        kind = _KINDS.get(shape)
        if kind is None:
            kind = _KINDS[shape] = kind_of(c)
        out.append(Comp(c.color, shape, kind, c.x1, c.y1, c.width, c.height, c.area, rel, c.click_point()))
    return out


def _step_cost(dx: int, dy: int) -> float:
    """Cost of explaining a component's change of place by a move of (dx, dy):
    path length, so one long jump (a ring wrapping round) is not traded for a
    reshuffle of many neighbours, plus a small quadratic term that breaks ties
    in favour of uniform shifts (a row of identical tiles moving one cell
    rather than one tile jumping two)."""
    return float(np.hypot(dx, dy)) + 0.01 * (dx * dx + dy * dy)


def match(before: list[Comp], after: list[Comp], by_kind: bool = True) -> dict[int, tuple[int, int]]:
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
        if len(b) == len(a) and sorted((c.x, c.y) for c in b) == sorted((c.x, c.y) for c in a):
            at = {(c.x, c.y): c.id for c in b}  # nothing of this kind changed place
            for q in a:
                q.id = at[(q.x, q.y)]
            continue
        px = np.array([c.x for c in b], dtype=float)[:, None]
        py = np.array([c.y for c in b], dtype=float)[:, None]
        dx = np.array([c.x for c in a], dtype=float)[None, :] - px
        dy = np.array([c.y for c in a], dtype=float)[None, :] - py
        cost = np.hypot(dx, dy) + 0.01 * (dx * dx + dy * dy)
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
    if not by_kind:
        return moved
    # A piece that turned as it moved (an avatar facing its way) changed its exact
    # shape but not its kind: pair what is left by colour and kind, close by only.
    taken = {c.id for c in after if c.id >= 0}
    left_b = [c for c in before if c.id not in taken]
    left_a = [c for c in after if c.id < 0]
    for p in left_b:
        best = None
        for q in left_a:
            if q.id >= 0 or q.color != p.color or q.kind != p.kind:
                continue
            dist = abs(q.x - p.x) + abs(q.y - p.y)
            if dist <= 12 and (best is None or dist < best[0]):
                best = (dist, q)
        if best is not None:
            q = best[1]
            q.id = p.id
            # the move of a turning piece is that of its bounding-box centre
            dx = round((q.x + (q.w - 1) / 2) - (p.x + (p.w - 1) / 2))
            dy = round((q.y + (q.h - 1) / 2) - (p.y + (p.h - 1) / 2))
            if (dx, dy) != (0, 0):
                moved[p.id] = (dx, dy)
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

    ``d_by`` keeps each moved component's own displacement (a mirror image that
    moves opposite to its piece), carried on beyond the places seen only for a
    shape every one of whose observed places moved by that same displacement, so
    a ring of tiles turning round its corners is never extrapolated.
    """

    cands: dict[tuple, set] = field(default_factory=dict)  # (shape, x, y) -> {(x, y)}
    best: dict[tuple, tuple[int, int]] = field(default_factory=dict)  # the current guess
    dirty: set[str] = field(default_factory=set)  # shapes whose guess must be recomputed
    _moving: set[str] | None = None
    _options: dict = field(default_factory=dict)
    movers: set[int] = field(default_factory=set)  # ids it moved last time
    d_by: dict[int, tuple[int, int]] = field(default_factory=dict)  # id -> its displacement last time
    moved_colors: set[int] = field(default_factory=set)  # colours of everything it has ever moved
    _offsets: dict[str, set] | None = None
    d: tuple[int, int] | None = None  # the displacement it gave most of them
    stale: bool = False  # a selection change since: re-bind on the next observed move
    uniform: bool = True  # every component it moved moved the same way (a translation)
    mode: bool = False  # it changed the frame without moving anything: a selection change
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
            self._moving = None
            self._offsets = None
            self._options = {}

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
        self._moving = None
        self._offsets = None
        self._options = {}
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
        d = self.d_by.get(cid)
        if d is not None and not self.stale and self.shape_offset(c_shape) == d:
            return (pos[0] + d[0], pos[1] + d[1])
        return pos

    def shape_offset(self, shape: str) -> tuple[int, int] | None:
        """The one displacement every observed place of ``shape`` moved by, if
        there is exactly one."""
        if self._offsets is None:
            offs: dict[str, set] = {}
            for k, v in self.cands.items():
                if len(v) == 1:
                    q = next(iter(v))
                    d = (q[0] - k[1], q[1] - k[2])
                    if d != (0, 0):
                        offs.setdefault(k[0], set()).add(d)
            self._offsets = offs
        o = self._offsets.get(shape)
        return next(iter(o)) if o and len(o) == 1 else None

    def options(self, c_shape: str, cid: int, pos: tuple[int, int], n: int = 3) -> list[tuple[int, int]]:
        """The guess and the next nearest candidates: what the action may do here.
        A place with many candidates is not worth enumerating: the guess only."""
        # keyed on the binding too: a translation that goes stale or is re-bound
        # changes what the action does at a place
        key = (c_shape, cid, pos, self.stale, self.d, self.uniform, cid in self.movers, self.d_by.get(cid))
        hit = self._options.get(key)
        if hit is not None:
            return hit
        first = self.predict(c_shape, cid, pos)
        c = self.cands.get((c_shape, pos[0], pos[1]))
        if not c or len(c) > MAX_OPTIONS:
            out = [first]
        else:
            near = sorted(c, key=lambda q: ((q[0] - pos[0]) ** 2 + (q[1] - pos[1]) ** 2, q))
            out = [first] + [q for q in near if q != first][: n - 1]
        self._options[key] = out
        return out

    def moving_shapes(self) -> set[str]:
        """Shapes this action has been seen to move somewhere."""
        if self._moving is None:
            self._moving = {k[0] for k, v in self.cands.items() if v != {k[1:]}}
        return self._moving

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


def exact_cover(slots: list[tuple[tuple[int, int], set]]) -> list[tuple[tuple[int, int], set]]:
    """The slots whose pixels do not overlap and together use every slot pixel:
    marks drawn side by side on a lattice also form spurious slots out of their
    neighbours' dots, which an exact cover leaves out. All slots when none exists."""
    if len(slots) <= 1:
        return slots
    universe = set().union(*(slot[1] for slot in slots))
    order = sorted(range(len(slots)), key=lambda i: slots[i][0])
    best: list[int] | None = None
    steps = 0

    def search(chosen: list[int], used: set) -> bool:
        nonlocal best, steps
        steps += 1
        if steps > 5000:
            return True
        if used == universe:
            best = list(chosen)
            return True
        first = min(universe - used)
        for i in order:
            pix = slots[i][1]
            if first in pix and not (pix & used):
                chosen.append(i)
                if search(chosen, used | pix):
                    return True
                chosen.pop()
        return False

    search([], set())
    return [slots[i] for i in best] if best else slots


def touching(a: Comp, b: Comp) -> bool:
    """Are two components 8-neighbours somewhere?"""
    if a.x - 1 > b.x + b.w - 1 or b.x - 1 > a.x + a.w - 1 or a.y - 1 > b.y + b.h - 1 or b.y - 1 > a.y + a.h - 1:
        return False
    pa = {(a.y + dy, a.x + dx) for dy, dx in a.rel}
    for dy, dx in b.rel:
        y, x = b.y + dy, b.x + dx
        for ny in (y - 1, y, y + 1):
            for nx in (x - 1, x, x + 1):
                if (ny, nx) in pa:
                    return True
    return False


def cover_places(rel: frozenset, pt: tuple[float, float]) -> set[tuple[int, int]]:
    """Every top-left position at which a component with pixel offsets ``rel``
    covers the point ``pt`` (see ``covers``)."""
    cx, cy = pt
    out = set()
    for x in {int(np.floor(cx)), int(np.ceil(cx))}:
        for y in {int(np.floor(cy)), int(np.ceil(cy))}:
            if abs(x - cx) <= 0.5 and abs(y - cy) <= 0.5:
                out.update((x - dx, y - dy) for dy, dx in rel)
    return out


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
    # Phase 26: the avatar as a piece. Goals of another colour learned from where a
    # moved piece ended, walkable and blocking colours for translated pieces, and
    # pieces that turn as they move matched by kind.
    navigation = True

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.confirmed = False  # it completed a level of this game itself
        self.disabled = False
        self.strikes = 0
        self.control_kinds: Counter = Counter()  # click-target kinds that moved a mover
        self.key_ops: Counter = Counter()  # keyboard actions that moved a mover
        self.prior_colors: set[int] = set()  # colours of the marks of earlier levels
        # How the pixels of a mark sat around the piece that covered it when a level
        # ended, by the piece's size: (dx, dy) of each mark pixel from its top-left.
        self.slot_patterns: dict[tuple[int, int], tuple[frozenset, bool, int | None]] = {}
        # A mark that sat level with the piece along one axis, far from it, when a level
        # ended (a token beside its patch): by the piece's size, (axis, offset of the
        # mark's centre from the piece's along that axis); axis 0 is x, 1 is y.
        self.alignments: dict[tuple[int, int], tuple[int, float]] = {}
        # Colours a translated piece has stepped onto, and colours that stopped it.
        self.entered: dict[int, set[int]] = {}
        self.blocked: dict[int, set[int]] = {}
        self.in_control = False
        self.stats: Counter = Counter()
        self._next_id = 0
        self.version = 0
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
        self.plan: list[tuple[OpKey, dict]] = []  # (action, where the search expects the pieces after it)
        self.expect: dict[int, tuple[int, int]] | None = None
        self.gave_up = False
        self.level_actions = 0
        self._level_work = 0
        self.last_key: OpKey | None = None
        self.mark_clusters_now: list[list[Comp]] = []
        self.exact_slots: dict[tuple, tuple[int, int, tuple[int, int]]] = {}
        self.align_slots: dict[tuple, tuple[int, int, int, int]] = {}  # (colour, point) -> (w, h, axis, top-left coordinate)
        self.mark_boxes: list[tuple] = []
        self.ghosts: dict[int, Comp] = {}  # movers merged into a same-coloured mark, at their place
        self.mode_presses = 0
        self._groups_cache = None
        self._failed_at: tuple | None = None
        self._idle = 0
        self._ops_seen = 0
        self._fails = 0
        self._budget = PLAN_BUDGET
        self.deaths = 0
        self.mark_ids: set[int] = set()
        self.grid = grid
        self.first_grid = grid
        self.bg = int(np.bincount(grid.ravel()).argmax()) if grid is not None else -1
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
        self._groups_cache = None
        if self.disabled:
            return
        if obs.state == GameState.GAME_OVER and self.in_control:
            self.deaths += 1
            self.stats["deaths"] += 1
        if obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        grid = np.asarray(obs.frame[-1])
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0:
                if len(obs.frame) > 1:
                    # The first frame of a level-up still shows the completed level.
                    self._learn_relations(np.asarray(obs.frame[0]))
                elif self.in_control:
                    self._remember_slots(action, data)
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
            self.grid = grid
            self.ghosts = {}
            self.plan = []
            self.expect = None
            return
        if {c.key for c in self.comps} != {c.key for c in after}:
            self.version += 1
        moved = match(self.comps, after, by_kind=self.navigation)
        # A new component touching a moved one of the same colour is part of the
        # same piece whose outline changed (an arm drawn shorter where it passes
        # behind something): it moved with it.
        if moved and len(after) >= 10:
            # The camera scrolled when most of the scene moved alike, counted in
            # components and in pixels (a line carrying many small dots is not the scene).
            common_d, n_common = Counter(moved.values()).most_common(1)[0]
            area = {c.id: c.area for c in after}
            alike = sum(area.get(cid, 0) for cid, d in moved.items() if d == common_d)
            if n_common >= SCROLL_SHARE * len(after) and alike >= SCROLL_SHARE * sum(area.values()):
                moved = {}  # nothing moved relative to the scene
                self.stats["scrolls"] += 1
        self._drop_occlusions(moved, after)
        moved_comps = [c for c in after if c.id in moved]
        for c in after:
            if c.id >= 0:
                continue
            c.id = self._new_id()
            for m in moved_comps:
                if m.color == c.color and touching(m, c):
                    moved[c.id] = moved[m.id]
                    break
        key = self.op_key(action, data)
        self.last_key = key
        op = self.ops.setdefault(key, Op())
        op.tried += 1
        before = {c.id: c for c in self.comps}
        after_by = {c.id: c for c in after}
        # A mark is never where a piece went (an arm tip merging into its mark leaves
        # identical dots elsewhere that would read as its destination).
        op.learn([c for c in self.comps if c.id not in self.mark_ids], [c for c in after if c.id not in self.mark_ids])
        if self.grid is not None and moved and len(moved) <= 4:
            # A piece translated onto a place: the colours it now covers are walkable for it.
            for cid, d in moved.items():
                b = before.get(cid)
                if b is None:
                    continue
                own = {(b.x + dx, b.y + dy) for dy, dx in b.rel}
                cols = {int(self.grid[b.y + dy + d[1], b.x + dx + d[0]]) for dy, dx in b.rel
                        if (b.x + dx + d[0], b.y + dy + d[1]) not in own
                        and 0 <= b.y + dy + d[1] < fp.GRID and 0 <= b.x + dx + d[0] < fp.GRID}
                self.entered.setdefault(b.color, set()).update(cols)
        if moved:
            for cid, d in moved.items():
                b = before.get(cid) or after_by[cid]
                self.movers.add(cid)
                self.mover_shapes.add((b.color, b.shape))
            common = Counter(moved.values()).most_common(1)[0][0]
            ids = {cid for cid, d in moved.items() if d == common}
            mine = {(before.get(cid) or after_by[cid]).color for cid in ids}
            alike = {d for cid, d in moved.items() if (before.get(cid) or after_by[cid]).color in mine}
            for other in self.ops.values():
                if other.stale and other.d is not None and other is not op:
                    other.movers = set(ids)  # the selection moved: re-bind
                    other.d_by = {}
                    other.stale = False
            op.movers, op.d, op.stale = set(ids), common, False
            op.d_by = dict(moved)
            op.uniform = op.uniform and len(alike) == 1
            # An action that has only ever moved things of colours no mark asks for
            # (selection dots passing from one piece to another) changes the selection:
            # what the other actions do must be seen again.
            op.moved_colors |= {(before.get(cid) or after_by[cid]).color for cid in moved}
            mark_cols = {col for col, _ in self._groups()[0]}
            if mark_cols and not op.moved_colors & mark_cols and not op.mode:
                op.mode = True
                for other in self.ops.values():
                    if other is not op:
                        other.stale = True
        # A mover that vanished where it was expected, its pixels still showing its
        # colour, has merged with a same-coloured mark (an arm tip entering a
        # cross): keep it there. A ghost whose pixels change colour is gone.
        now = {c.id: (c.x, c.y) for c in after}
        for cid, g in list(self.ghosts.items()):
            if not self._shows(grid, g, (g.x, g.y)):
                del self.ghosts[cid]
        if self.expect is not None:
            for cid, pos in self.expect.items():
                b = before.get(cid) or self.ghosts.get(cid)
                if b is not None and cid not in now and self._shows(grid, b, pos):
                    g = Comp(b.color, b.shape, b.kind, pos[0], pos[1], b.w, b.h, b.area, b.rel, b.point, cid)
                    self.ghosts[cid] = g
                    now[cid] = pos
        # Movers that vanished without being explained (moved, merged): their shape
        # changed, as when a selection marker is drawn on or taken off a piece. The
        # actions bound to them go stale; an action that did only that is a
        # selection change.
        lost = (set(before) - set(now)) & self.movers
        lost -= set(self.ghosts)
        if lost:
            lost_cols = {before[cid].color for cid in lost}
            same = {cid for cid in moved if after_by[cid].color in lost_cols}
            for other in self.ops.values():
                if not other.movers & lost:
                    continue
                if same:  # the piece moved on with a changed outline: re-bind
                    other.movers = (other.movers - lost) | same
                else:
                    other.stale = True
                for cid in lost:
                    other.d_by.pop(cid, None)
            if not same:
                op.mode = True
        # A relevant mover the model expected to move but which stayed: blocked here.
        if self.expect is not None:
            wrong = False
            for cid, pos in self.expect.items():
                b = before.get(cid) or self.ghosts.get(cid)
                if b is None or now.get(cid) == pos:
                    continue
                wrong = True
                if now.get(cid) == (b.x, b.y):
                    op.block(b.shape, (b.x, b.y))  # it stayed: blocked here
                    if self.grid is not None:
                        own = {(b.x + dx, b.y + dy) for dy, dx in b.rel}
                        cols = {int(self.grid[pos[1] + dy, pos[0] + dx]) for dy, dx in b.rel
                                if (pos[0] + dx, pos[1] + dy) not in own
                                and 0 <= pos[1] + dy < fp.GRID and 0 <= pos[0] + dx < fp.GRID}
                        walk = self.entered.get(b.color, set())
                        self.blocked.setdefault(b.color, set()).update(cols - walk)
            if wrong:
                self.mismatches += 1
                self.stats["mismatches"] += 1
                self.plan = []
            self.expect = None
        self.comps = after
        self.grid = grid
        self._groups_cache = None  # it may have been computed from the frame before

    def _drop_occlusions(self, moved: dict[int, tuple[int, int]], after: list[Comp]) -> None:
        """A piece walking over scenery hides the cell it steps on and uncovers the
        one it left, which reads as that cell moving the opposite way. Of two
        components that swapped places with opposite moves, the one common in the
        frame (a floor cell among many) did not move."""
        if not moved:
            return
        before = {c.id: c for c in self.comps}
        now = {c.id: c for c in after}
        count = Counter((c.color, c.shape) for c in after)
        # One of many identical cells changing place is a cell hidden and another
        # uncovered by something passing over the floor, not a cell that moved.
        by_class: dict[tuple, list[int]] = {}
        for cid in moved:
            if cid in now:
                by_class.setdefault((now[cid].color, now[cid].shape), []).append(cid)
        for cls, cids in by_class.items():
            if len(cids) == 1 and count[cls] >= SCENERY_COPIES:
                del moved[cids[0]]
                self.stats["occlusions"] += 1
        if len(moved) < 2:
            return

        def box(c: Comp) -> tuple[int, int, int, int]:
            return (c.x, c.y, c.x + c.w - 1, c.y + c.h - 1)

        def overlap(a, b) -> bool:
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        ids = list(moved)
        drop = set()
        for i, x in enumerate(ids):
            for y in ids[i + 1:]:
                dx, dy = moved[x], moved[y]
                if (dx[0] + dy[0], dx[1] + dy[1]) != (0, 0):
                    continue
                if x not in before or y not in before or x not in now or y not in now:
                    continue
                if overlap(box(now[x]), box(before[y])) and overlap(box(before[x]), box(now[y])):
                    cx, cy = now[x], now[y]
                    common = x if count[(cx.color, cx.shape)] >= count[(cy.color, cy.shape)] else y
                    if count[(now[common].color, now[common].shape)] > 1:
                        drop.add(common)
        for cid in drop:
            del moved[cid]
            self.stats["occlusions"] += 1

    def _passable(self, m: Comp, pos: tuple[int, int], q: tuple[int, int]) -> bool:
        """Can piece ``m`` be translated from ``pos`` to ``q``, by the colours it
        has stepped onto and been stopped by? Unknown colours are tried; the
        place half-way is checked too (a piece that jumps a cell is stopped by a
        wall between)."""
        stop = self.blocked.get(m.color)
        if not self.navigation or not stop or self.grid is None:
            return True
        walk = self.entered.get(m.color, set())
        own = {(m.x + dx, m.y + dy) for dy, dx in m.rel}
        mid = ((pos[0] + q[0]) // 2, (pos[1] + q[1]) // 2)
        for at in (q, mid):
            for dy, dx in m.rel:
                x, y = at[0] + dx, at[1] + dy
                if not (0 <= x < fp.GRID and 0 <= y < fp.GRID):
                    return False
                if (x, y) in own:
                    continue
                c = int(self.grid[y, x])
                if c in stop and c not in walk:
                    return False
        return True

    def _shows(self, grid: np.ndarray, c: Comp, pos: tuple[int, int]) -> bool:
        """Is ``c`` at ``pos``: do its pixels show its colour, but for pixels hidden
        under something drawn there since the level began (a target outline or a
        line drawn over the piece), and at least half of them?"""
        h, w = grid.shape
        shown = 0
        for dy, dx in c.rel:
            y, x = pos[1] + dy, pos[0] + dx
            if not (0 <= y < h and 0 <= x < w):
                return False
            v = int(grid[y, x])
            if v == c.color:
                shown += 1
            elif self.first_grid is None or v != int(self.first_grid[y, x]) or v == self.bg:
                return False
        return 2 * shown >= len(c.rel)

    def _remember_slots(self, action, data) -> None:
        """The last action completed the level: the marks were covered right after
        it. Predict where the movers went and record, for each mark cluster a mover
        covered, its pixels relative to the mover."""
        if self.last_key is None or self.last_key not in self.ops:
            return
        op = self.ops[self.last_key]
        groups, movers = self._groups()
        placed = [(m, op.predict(m.shape, m.id, (m.x, m.y))) for m in movers]
        for cl in self.mark_clusters_now:
            pt = centre(cl)
            for m, pos in placed:
                if m.color == cl[0].color and covers(m.rel, pos, pt):
                    pix = {(c.x + dx - pos[0], c.y + dy - pos[1]) for c in cl for dy, dx in c.rel}
                    self.slot_patterns[(m.w, m.h)] = (frozenset(pix), True, None)
                    self.stats["slot_patterns"] += 1
                    break

    def _learn_relations(self, final_grid: np.ndarray) -> None:
        """How each same-coloured mark sat relative to the nearest moving piece in
        the completed level's last frame: on it (lp85's tile inside its corners,
        a stroke over a dot) or beside it (vc33's token level with its patch).
        Recorded by the piece's size for the next levels."""
        if not self.mover_shapes:
            return
        final = components(final_grid, self.mask)
        match([Comp(c.color, c.shape, c.kind, c.x, c.y, c.w, c.h, c.area, c.rel, c.point, c.id)
               for c in self.comps], final)
        mover_colors = {col for col, _ in self.mover_shapes}
        movers = [c for c in final if c.color in mover_colors
                  and (c.id in self.movers or (c.color, c.shape) in self.mover_shapes)]
        static = [c for c in self.first if c.id not in self.movers and c.color in mover_colors]
        learned = False
        for cl in mark_clusters(static):
            col = cl[0].color
            if len(cl) == 1 and (col, cl[0].shape) in self.mover_shapes:
                continue
            near = [m for m in movers if m.color == col]
            if not near:
                continue
            pt = centre(cl)

            def gap(m: Comp) -> float:
                return abs(m.x + (m.w - 1) / 2 - pt[0]) + abs(m.y + (m.h - 1) / 2 - pt[1])

            m = min(near, key=gap)
            if gap(m) > max(m.w, m.h) + 6:
                # Nothing of its colour near it; but a piece level with it along one
                # axis (a token beside the patch of its colour) is a relation too.
                mc = (m.x + (m.w - 1) / 2, m.y + (m.h - 1) / 2)
                for axis in (0, 1):
                    if abs(mc[axis] - pt[axis]) <= 0.5:
                        self.alignments[(m.w, m.h)] = (axis, pt[axis] - mc[axis])
                        self.stats["alignments"] += 1
                        learned = True
                        break
                continue
            pix = frozenset((c.x + dx - m.x, c.y + dy - m.y) for c in cl for dy, dx in c.rel)
            self.slot_patterns[(m.w, m.h)] = (pix, covers(m.rel, (m.x, m.y), pt), None)
            self.stats["relations"] += 1
            learned = True
        if not learned and self.navigation:
            self._learn_cross_relation(final)

    def _learn_cross_relation(self, final: list[Comp]) -> None:
        """No mark of a mover's own colour: the goal may be an object of another
        colour the piece ended on (tu93's avatar on its exit). Record the small
        static object the moved piece overlaps most in the completed level's
        last frame."""
        count = Counter((c.color, c.shape) for c in self.first)
        static = [c for c in self.first if c.id not in self.movers and c.area <= 64
                  and count[(c.color, c.shape)] < SCENERY_COPIES
                  and (c.color, c.shape) not in self.mover_shapes]
        best = None
        for m in final:
            if m.id not in self.movers:
                continue
            mine = {(m.x + dx, m.y + dy) for dy, dx in m.rel}
            for c in static:
                if c.color == m.color:
                    continue
                theirs = {(c.x + dx, c.y + dy) for dy, dx in c.rel}
                n = len(mine & theirs)
                if n and (best is None or n > best[0]):
                    best = (n, m, c, theirs)
        if best is None:
            return
        _, m, c, theirs = best
        pix = frozenset((x - m.x, y - m.y) for x, y in theirs)
        self.slot_patterns[(m.w, m.h)] = (pix, covers(m.rel, (m.x, m.y), centre([c])), c.color)
        self.stats["cross_relations"] += 1

    def _pattern_colour(self, w: int, h: int, color: int) -> int:
        """The colour the marks of a w x h piece of ``color`` are drawn in."""
        entry = self.slot_patterns.get((w, h))
        if entry is None and self.slot_patterns:
            entry = min(self.slot_patterns.items(), key=lambda kv: abs(kv[0][0] - w) + abs(kv[0][1] - h))[1]
        return color if entry is None or entry[2] is None else entry[2]

    def _slots(self, color: int, w: int, h: int, static: set) -> list[tuple[tuple[int, int], set, bool]]:
        """Placements of a w x h piece around which a learned mark pattern is drawn in
        ``color``: (top-left, mark pixels)."""
        entry = self.slot_patterns.get((w, h))
        pat, cover, _ = entry if entry is not None else (None, True, None)
        if pat is None and self.slot_patterns:
            (pw, ph), (base, cover, _) = min(self.slot_patterns.items(),
                                             key=lambda kv: abs(kv[0][0] - w) + abs(kv[0][1] - h))
            if pw and w % pw == 0 and h % ph == 0 and w // pw == h // ph:
                k = w // pw  # the level is drawn k times larger: scale the pattern
                pat = frozenset((dx * k + i, dy * k + j) for dx, dy in base for i in range(k) for j in range(k))
            elif w and pw % w == 0 and ph % h == 0 and pw // w == ph // h:
                k = pw // w  # drawn k times smaller
                pat = frozenset((dx // k, dy // k) for dx, dy in base if dx % k == 0 and dy % k == 0)
        if not pat:
            return []
        anchor = min(pat)
        out = []
        for (x, y) in static:
            p = (x - anchor[0], y - anchor[1])
            pix = {(p[0] + dx, p[1] + dy) for dx, dy in pat}
            if pix <= static:
                out.append((p, pix, cover))
        return out

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
        if self._groups_cache is not None:
            return self._groups_cache
        self._groups_cache = self._compute_groups()
        return self._groups_cache

    def _compute_groups(self) -> tuple[list[tuple[int, list[tuple[float, float]]]], list[Comp]]:
        mover_colors = {col for col, _ in self.mover_shapes}
        marks: dict[int, list[tuple[float, float]]] = {}
        static = [c for c in self.first if c.id not in self.movers and c.color in mover_colors]
        self.mark_boxes = []
        self.mark_clusters_now = []
        self.exact_slots = {}
        self.align_slots = {}
        slot_ids: set[int] = set()
        if self.slot_patterns:
            # A mark drawn the way earlier levels' marks were: one slot per placement,
            # so marks drawn side by side on a lattice stay apart.
            sizes = {(c.color, c.w, c.h) for c in self.comps if (c.color, c.shape) in self.mover_shapes}
            still = [c for c in self.first if c.id not in self.movers]
            for (col, w, h) in sorted(sizes):
                mcol = self._pattern_colour(w, h, col)
                pix_owner = {}
                for c in still:
                    if c.color == mcol:
                        for dy, dx in c.rel:
                            pix_owner[(c.x + dx, c.y + dy)] = c
                if True:
                    for pos, pix, cover in exact_cover(self._slots(col, w, h, set(pix_owner))):
                        xs = [q[0] for q in pix]; ys = [q[1] for q in pix]
                        if cover:
                            pt = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
                        else:
                            pt = (pos[0] + (w - 1) / 2.0, pos[1] + (h - 1) / 2.0)
                            self.exact_slots[(col, pt)] = (w, h, pos)
                        marks.setdefault(col, []).append(pt)
                        self.mark_boxes.append((col, min(xs), min(ys), max(xs), max(ys)))
                        owners = {pix_owner[q].id: pix_owner[q] for q in pix}
                        slot_ids |= set(owners)
                        self.mark_clusters_now.append(list(owners.values()))
        if not marks:
            # pieces the level began with that have moved (a fragment of a piece
            # uncovered for a moment is not one)
            first_movers = {(c.color, c.shape) for c in self.first if c.id in self.movers}
            for cl in mark_clusters(static):
                if len(cl) == 1 and (cl[0].color, cl[0].shape) in first_movers:
                    continue  # looks like a piece that has not moved yet
                marks.setdefault(cl[0].color, []).append(centre(cl))
                self.mark_clusters_now.append(cl)
                self.mark_boxes.append((cl[0].color, min(c.x for c in cl), min(c.y for c in cl),
                                        max(c.x + c.w - 1 for c in cl), max(c.y + c.h - 1 for c in cl)))
        if marks and self.alignments:
            # marks met by standing level with them along an axis
            sizes: dict[int, set] = {}
            for c in self.comps:
                if (c.color, c.shape) in self.mover_shapes:
                    sizes.setdefault(c.color, set()).add((c.w, c.h))
            for col, pts in marks.items():
                for w, h in sorted(sizes.get(col, ())):
                    al = self.alignments.get((w, h))
                    if al is None:
                        continue
                    axis, off = al
                    for pt in pts:
                        v = pt[axis] - off - ((w if axis == 0 else h) - 1) / 2
                        if abs(v - round(v)) < 0.25:
                            self.align_slots[(col, pt)] = (w, h, axis, int(round(v)))
                    break
        if not marks:
            # No mark of a moving colour: a still shape exactly like a moving piece's,
            # drawn in a colour no piece has, is where that piece goes (an outline of
            # a piece, or of its mirror image, to be covered by it). Not for a plain
            # block, whose look-alikes (walls, tiles) are everywhere, nor for a copy of
            # the whole sprite, small parts included (an enemy drawn like the avatar).
            count = Counter((c.color, c.shape) for c in self.first)
            by_shape: dict[str, list[Comp]] = {}
            for c in self.comps:
                if c.id in self.movers:
                    by_shape.setdefault(c.shape, []).append(c)

            def parts(c: Comp, comps: list[Comp]) -> set:
                return {(a.shape, a.x - c.x, a.y - c.y) for a in comps
                        if a.color != c.color and a.area < c.area and touching(a, c)}
            for c in self.first:
                if (c.id in self.movers or c.color in mover_colors or c.area < 4
                        or count[(c.color, c.shape)] >= SCENERY_COPIES):
                    continue
                m = next(iter(by_shape.get(c.shape, [])), None)
                if m is None or m.area == m.w * m.h:
                    continue
                mine = parts(m, self.comps)
                if mine and mine <= parts(c, self.first):
                    continue
                pt = (c.x + (c.w - 1) / 2.0, c.y + (c.h - 1) / 2.0)
                marks.setdefault(m.color, []).append(pt)
                self.exact_slots[(m.color, pt)] = (m.w, m.h, (c.x, c.y))
                self.mark_clusters_now.append([c])
                self.mark_boxes.append((m.color, c.x, c.y, c.x + c.w - 1, c.y + c.h - 1))
                self.stats["silhouettes"] += 1
        groups = [(col, pts) for col, pts in sorted(marks.items())]
        self.mark_ids = {c.id for cl in self.mark_clusters_now for c in cl}
        colors = {col for col, _ in groups}
        # Movers of those colours: components seen moving, and look-alikes that have
        # not moved yet (a second goal tile on a ring not turned yet), but not the
        # pieces of a mark.
        movers = [c for c in self.comps if c.color in colors and c.id not in self.mark_ids
                  and (c.id in self.movers or (c.color, c.shape) in self.mover_shapes)]
        movers += [g for g in self.ghosts.values() if g.color in colors]
        return groups, movers

    def _frame_covered(self, col: int, pt: tuple[float, float]) -> Comp | None:
        """The current component (not part of a mark) of colour ``col`` covering ``pt``,
        or for a side-by-side mark, standing in its place."""
        slot = self.exact_slots.get((col, pt))
        if slot is not None:
            w, h, pos = slot
            for c in list(self.comps) + list(self.ghosts.values()):
                if c.color == col and (c.w, c.h) == (w, h) and (c.x, c.y) == pos and c.id not in self.mark_ids:
                    return c
            return None
        al = self.align_slots.get((col, pt))
        if al is not None:
            w, h, axis, v = al
            for c in list(self.comps) + list(self.ghosts.values()):
                if c.color == col and (c.w, c.h) == (w, h) and (c.x, c.y)[axis] == v and c.id not in self.mark_ids:
                    return c
            return None
        for c in list(self.comps) + list(self.ghosts.values()):
            if c.color != col or c.id in self.mark_ids or not covers(c.rel, (c.x, c.y), pt):
                continue
            if c.id not in self.movers and c.id not in self.ghosts and any(
                    bc == col and bx1 <= c.x and by1 <= c.y and c.x + c.w - 1 <= bx2 and c.y + c.h - 1 <= by2
                    for bc, bx1, by1, bx2, by2 in self.mark_boxes):
                continue  # a piece of the mark itself, seen again after something passed over it
            return c
        return None

    def recognised(self) -> bool:
        return bool(self.mover_shapes) and bool(self._groups()[0])

    # -- planning ---------------------------------------------------------------

    def _satisfied(self, groups, movers, state) -> list[bool]:
        out = []
        for col, pts in groups:
            idx = [i for i, m in enumerate(movers) if m.color == col]
            out.append(all(any(covers(movers[i].rel, state[i], p) for i in idx) for p in pts))
        return out

    def _search(self, groups, movers, need, optimistic: bool = False,
                limit: int | None = None) -> list[tuple[OpKey, dict]] | None:
        """Shortest op sequence after which the groups flagged in ``need`` are
        satisfied, each op with the piece positions the search expects after it.
        ``optimistic``: where the model is still unsure what an action does at a
        place, any of the nearest candidates may happen; running such a plan is
        also how the uncertainty gets resolved."""
        live_shapes = {m.shape for m in movers}
        live_ids = {m.id for m in movers} | {m.id for m in self.ghosts.values()}
        # Undo (ACTION7) depends on the history, not on where the pieces are: never a step.
        ops = [(k, op) for k, op in self.ops.items() if k != ("k", 7)
               and ((op.d is not None and not op.stale and op.uniform and op.movers & live_ids)
                    or op.moving_shapes() & live_shapes)]
        # Pieces no action can move from where they are stay out of the search state.
        fixed = [m for m in movers if all(op.predict(m.shape, m.id, (m.x, m.y)) == (m.x, m.y)
                                          and (not optimistic or op.options(m.shape, m.id, (m.x, m.y)) == [(m.x, m.y)])
                                          for _, op in ops)]
        fixed_ids = {m.id for m in fixed}
        live = [m for m in movers if m.id not in fixed_ids]
        everyone = live + fixed
        rest = tuple((m.x, m.y) for m in fixed)
        start = tuple((m.x, m.y) for m in live)
        if limit is None:
            limit = MAX_EXPANSIONS if not optimistic else MAX_EXPANSIONS // 5

        def inside(m: Comp, q: tuple[int, int]) -> bool:
            return -m.w < q[0] < fp.GRID and -m.h < q[1] < fp.GRID
        # For every needed mark, the (piece, top-left) pairs at which it is covered;
        # marks a fixed piece already covers need nothing.
        needs: list[list[tuple[int, set]]] = []
        live_ids = {m.id for m in live}
        wanted = need if isinstance(need, set) else None
        for (col, pts), n in zip(groups, need if wanted is None else [True] * len(groups)):
            if not n:
                continue
            for pt in pts:
                if wanted is not None and (col, pt) not in wanted:
                    continue
                c = self._frame_covered(col, pt)
                if c is not None and c.id not in live_ids:
                    continue  # covered by something no action moves
                slot = self.exact_slots.get((col, pt))
                al = self.align_slots.get((col, pt))
                if slot is not None:
                    w, h, pos = slot
                    needs.append([(i, {pos}) for i, m in enumerate(live) if m.color == col and (m.w, m.h) == (w, h)])
                elif al is not None:
                    w, h, axis, v = al
                    line = {(v, y) for y in range(-h + 1, fp.GRID)} if axis == 0 else {(x, v) for x in range(-w + 1, fp.GRID)}
                    needs.append([(i, line) for i, m in enumerate(live) if m.color == col and (m.w, m.h) == (w, h)])
                else:
                    needs.append([(i, cover_places(m.rel, pt)) for i, m in enumerate(live) if m.color == col])
        if any(not opts for opts in needs):
            return None
        # Relaxation: every needed mark must be reachable by some piece moving alone.
        reach = []
        for m in live:
            seen_m = {(m.x, m.y)}
            frontier = deque([(m.x, m.y)])
            work = 0
            while frontier and len(seen_m) < 4000 and work < MAX_EXPANSIONS // 3:
                pos = frontier.popleft()
                work += len(ops)  # one piece: one prediction per action
                for _, op in ops:
                    nxts = op.options(m.shape, m.id, pos) if optimistic else [op.predict(m.shape, m.id, pos)]
                    for q in nxts:
                        if q not in seen_m and inside(m, q):
                            seen_m.add(q)
                            frontier.append(q)
            reach.append(seen_m)
        for opts in needs:
            if not any(reach[i] & places for i, places in opts):
                return None

        def done(state) -> bool:
            return all(any(state[i] in places for i, places in opts) for opts in needs)

        memo: dict[tuple, tuple[int, int]] = {}

        def predict(key, op, m, pos):
            k = (key, m.id, pos)
            hit = memo.get(k)
            if hit is None:
                hit = op.predict(m.shape, m.id, pos)
                if hit != pos and (m.shape, pos[0], pos[1]) not in op.cands and not self._passable(m, pos, hit):
                    hit = pos  # a translation into a colour that has stopped this piece
                memo[k] = hit
            return hit

        def successors(state, key, op):
            if not optimistic:
                yield tuple(predict(key, op, m, pos) for m, pos in zip(live, state))
                return
            choices = [op.options(m.shape, m.id, pos) for m, pos in zip(live, state)]
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
        limit = min(limit, self._budget)
        while queue and expansions < limit:
            state = queue.popleft()
            expansions += len(ops) * max(1, len(live))
            for key, op in ops:
                for nxt in successors(state, key, op):
                    if nxt in seen or not all(inside(m, q) for m, q in zip(live, nxt)):
                        continue
                    seen.add(nxt)
                    parent[nxt] = (state, key)
                    if done(nxt):
                        self._budget -= expansions
                        path = []
                        while nxt != start:
                            prev, k = parent[nxt]
                            path.append((k, nxt))
                            nxt = prev
                        stay = {m.id: (m.x, m.y) for m in fixed}
                        return [(k, {**stay, **{m.id: q for m, q in zip(live, st)}}) for k, st in reversed(path)]
                    queue.append(nxt)
        self._budget -= expansions
        return None

    def _make_plan(self) -> list[tuple[OpKey, dict]] | None:
        self._budget = PLAN_BUDGET
        try:
            return self._make_plan_inner()
        finally:
            self._level_work += PLAN_BUDGET - self._budget

    def _make_plan_inner(self) -> list[tuple[OpKey, dict]] | None:
        groups, movers = self._groups()
        if not groups:
            return None
        marks = [(col, pt) for col, pts in groups for pt in pts]
        covered = {m for m in marks if self._frame_covered(*m) is not None}
        if len(covered) == len(marks):
            return []
        everything = [True] * len(groups)
        plan = self._search(groups, movers, everything)
        if plan is None:
            plan = self._search(groups, movers, everything, optimistic=True)
            if plan is not None:
                self.stats["optimistic_plans"] += 1
        if plan is not None:
            return plan
        # Too much at once (several pieces, a decoy colour, a piece not selected
        # yet): one more mark at a time, keeping the covered ones covered if that
        # can be done, shortest first; else one mark alone.
        best = None
        for m in marks:
            if m in covered:
                continue
            p = self._search(groups, movers, covered | {m}, limit=MAX_EXPANSIONS // 4)
            if p is None:
                p = self._search(groups, movers, covered | {m}, optimistic=True, limit=MAX_EXPANSIONS // 8)
            if p and (best is None or len(p) < len(best)):
                best = p
        if best is None:
            for m in marks:
                if m in covered:
                    continue
                p = self._search(groups, movers, {m}, limit=MAX_EXPANSIONS // 4)
                if p and (best is None or len(p) < len(best)):
                    best = p
        if best is not None:
            self.stats["partial_plans"] += 1
        return best

    def _plan_if_new(self) -> list[tuple[OpKey, dict]] | None:
        """A plan, unless the search already failed on the same model and frame."""
        key = (self.version, self.level)
        if self._failed_at == key:
            return None
        plan = self._make_plan()
        if plan is None:
            self._failed_at = key
        return plan

    # -- probing ----------------------------------------------------------------

    def _probe(self, obs: FrameDataRaw, lookalike: bool = False) -> OpKey | None:
        """The next action to try. ``lookalike``: only click targets shaped like a
        control that has moved something (controls tend to come in families,
        such as the two arrows of a ring)."""
        if self.probes >= (MAX_PROBES if self.confirmed else UNPROVEN_PROBES):
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
            if op is None and k not in self.probed:
                return k
            if op is not None and op.stale:
                op.stale = False  # probed again now: re-bound by what it moves
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

    def _give_up(self, why: str = "") -> None:
        if why:
            self.stats["give_up_" + why] += 1
        self.gave_up = True
        self.in_control = False
        self.strikes += 1
        self.stats["give_ups"] += 1
        # A give-up costs the solver the rest of the level; the next level, with
        # what the completed one taught (its controls, how its marks were met),
        # gets another try. Two give-ups end it, three once it has solved a level.
        if self.strikes >= (3 if self.confirmed else 2):
            self.disabled = True

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not self.in_control:
            if self.confirmed and self.probes == 0 and not self.ops:
                pass  # a game this solver has solved levels of: take the new level from its start
            elif not self.recognised():
                return None
            elif not self.confirmed:
                # Not proven on this game yet: take over only with a plan in hand, so
                # a game that merely looks like the family costs nothing.
                # Failed searches back off (5, 10, 20, ... observations, up to 160)
                # unless a new action has been seen since.
                self._idle += 1
                wait = min(PLAN_EVERY << self._fails, 160)
                if len(self.movers) == self._ops_seen and self._idle < wait:
                    return None
                if self._level_work > LEVEL_WORK:
                    return None  # this level has had its share of searching
                self._idle = 0
                if len(self.movers) != self._ops_seen:
                    self._fails = 0
                self._ops_seen = len(self.movers)
                plan = self._plan_if_new()
                if not plan:
                    self._fails += 1
                if not plan:
                    return None
                self.plan = plan
                self.stats["plans"] += 1
            self.in_control = True
            self._fails = self._idle = 0  # a fresh take-over plans at once
            self.stats["takeovers"] += 1
        if self.mismatches > MAX_MISMATCHES:
            self._give_up("mismatches")
            return None
        if self.deaths >= MAX_DEATHS:
            self._give_up("deaths")
            return None
        if self.level_actions >= MAX_LEVEL_ACTIONS:
            self._give_up("actions")
            return None
        self.level_actions += 1
        if not self.plan:
            plan = None
            if self.recognised():
                self._idle += 1
                wait = min(1 << self._fails, 32)
                if len(self.movers) != self._ops_seen or self._idle >= wait:
                    if len(self.movers) != self._ops_seen:
                        self._fails = 0
                    self._ops_seen = len(self.movers)
                    self._idle = 0
                    plan = self._plan_if_new()
                    self._fails = 0 if plan is not None else self._fails + 1
            if plan == []:
                # Every known mark is covered and the level has not ended: another
                # piece, not moved yet, may have marks of its own. Keep probing.
                plan = None
                self.stats["covered_waits"] += 1
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
            # Every step of a plan changes the searched state: a step after which a
            # piece is not where the search put it (moved otherwise, or not at all)
            # is a misprediction.
            key, self.expect = self.plan.pop(0)
            self.stats["plan_steps"] += 1
            return self._action(key)
        key = self._probe(obs)
        if key is None and not self.recognised():
            # Nothing to aim at yet (no piece of a marked colour has moved): leave
            # the turn to the other agent without a strike and look again later.
            self.in_control = False
            self.stats["yields"] += 1
            return None
        if key is None:
            # Nothing left to probe and no plan: the piece that must move may not be
            # the selected one. Change the selection (then re-probe the moves).
            modes = [k for k, op in self.ops.items() if op.mode]
            groups, _ = self._groups()
            limit = 2 * len(groups) + 2 if self.confirmed else 2
            if modes and groups and self.mode_presses < limit:
                self.mode_presses += 1
                self.stats["mode_presses"] += 1
                return self._action(modes[0])
            self._give_up("no_probe")
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


class MarkerSolverNoNav(MarkerSolver):
    """The Phase 25 solver alone (no navigation additions): an ablation."""

    navigation = False
