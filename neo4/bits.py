"""The bit-program family: set a program's bits so that running it brings a piece onto its outline.

Some puzzles show a board with a piece and the piece's outline (where it must end,
turned, sized and coloured as the outline shows), and under the board a program panel:
columns of bits, each column one instruction, its value the bits that are on. Clicking a
bit flips it; a button runs the program, a column at a time, each moving, turning,
resizing or recolouring the piece; a piece that ends on its outline completes the level,
any other goes back to its start. Obstacles stand in the way. A second board may show
demonstrations: tabs under it each run a demo program there.

From the frames alone:

* a panel is a lattice of small glyphs of two colours (on, off) on a ground of one colour,
  the same glyph in every column of a row; its board is the area above it;
* the piece and its target are the two objects of the board one of which is the outline
  of the other (its ring, the piece turned and scaled); other objects of the piece's or the
  target's colour are marks, the rest obstacles;
* the button is the solid object off the boards nearest the panel; tabs are a row of
  equal square frames;
* what an instruction does is not known in advance: every run (a demo or the solver's
  own) is watched frame by frame, one frame per column, and the change of the piece in a
  frame is what that column's value does;
* a run that ends on a mark is expected to move the start there; that is checked.

The plan is the cheapest program: per column a value whose effect is known, at the cost
of the bits to flip from what the column holds now; a search over the columns from the
piece's start with the obstacles forbidden, and if no single run reaches the outline,
runs that end on a mark first. Values not yet seen are learned from the demos, tab by
tab, as the plan needs them, or by running what the panel holds.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - scipy is optional
    ndimage = None

Action = tuple[GameAction, dict[str, int] | None]

MAX_GLYPH = 6  # pixels of a bit glyph, at most
MAX_SCALE = 4
MAX_RUNS = 3  # runs chained through marks in one plan
MAX_FAILS = 4  # runs that did not go as planned, per level
MAX_LEVEL_ACTIONS = 120
CHECKER_CELL = 32  # the cells of a two-colour floor are at most this many pixels


# --- reading the screen ------------------------------------------------------------


@dataclass
class Glyph:
    x: int
    y: int  # a click point
    color: int
    shape: tuple


@dataclass
class Panel:
    ground: int
    cols: list  # columns left to right, each a list of glyphs top to bottom
    box: tuple  # x1, y1, x2, y2 of the glyphs

    def patterns(self) -> list[tuple]:
        return [tuple(g.color for g in col) for col in self.cols]


def _ground(grid: np.ndarray, c: fp.Component) -> int:
    H, W = grid.shape
    y1, y2, x1, x2 = max(c.y1 - 1, 0), min(c.y2 + 2, H), max(c.x1 - 1, 0), min(c.x2 + 2, W)
    box = grid[y1:y2, x1:x2]
    own = np.zeros(box.shape, dtype=bool)
    own[c.pixels[:, 0] - y1, c.pixels[:, 1] - x1] = True
    vals = box[~own]
    if vals.size == 0:
        return -1
    return int(np.bincount(vals.ravel()).argmax())


def _clusters(values: list[float], gap: float) -> list[float]:
    """Centres of runs of sorted values no more than gap apart."""
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(c) / len(c) for c in out]


def find_panels(grid: np.ndarray, comps: list) -> list[Panel]:
    groups: dict = {}
    for c in comps:
        if c.area > MAX_GLYPH or c.width > 3 or c.height > 3:
            continue
        g = _ground(grid, c)
        if g < 0 or g == c.color:
            continue
        shape = tuple(sorted(zip((c.pixels[:, 0] - c.y1).tolist(), (c.pixels[:, 1] - c.x1).tolist())))
        groups.setdefault(g, []).append((c, shape))
    panels = []
    for ground, items in groups.items():
        cxs = _clusters([(c.x1 + c.x2) / 2 for c, _ in items], 1.0)
        cys = _clusters([(c.y1 + c.y2) / 2 for c, _ in items], 1.0)
        if len(cxs) < 2 or len(cys) < 2:
            continue
        cell: dict = {}
        for c, shape in items:
            i = min(range(len(cxs)), key=lambda k: abs(cxs[k] - (c.x1 + c.x2) / 2))
            j = min(range(len(cys)), key=lambda k: abs(cys[k] - (c.y1 + c.y2) / 2))
            cell.setdefault((i, j), []).append((c, shape))
        # rows held by every column, each with one glyph of one shape throughout
        rows = [j for j in range(len(cys)) if all(len(cell.get((i, j), [])) == 1 for i in range(len(cxs)))]
        rows = [j for j in rows if len({cell[(i, j)][0][1] for i in range(len(cxs)) if True}) == 1]
        if len(rows) < 2:
            continue
        colours = {cell[(i, j)][0][0].color for i in range(len(cxs)) for j in rows}
        if len(colours) > 2:
            continue
        cols = []
        for i in range(len(cxs)):
            col = []
            for j in rows:
                c, shape = cell[(i, j)][0]
                x, y = c.click_point()
                col.append(Glyph(x, y, c.color, shape))
            cols.append(col)
        xs = [g.x for col in cols for g in col]
        ys = [g.y for col in cols for g in col]
        panels.append(Panel(ground, cols, (min(xs), min(ys), max(xs), max(ys))))
    return panels


def _blocky(mask: np.ndarray) -> int:
    """The largest s such that the mask is an s-fold enlargement of a smaller one."""
    h, w = mask.shape
    for s in range(MAX_SCALE, 1, -1):
        if h % s or w % s:
            continue
        small = mask[::s, ::s]
        if (np.kron(small, np.ones((s, s), dtype=bool)) == mask).all():
            return s
    return 1


@dataclass
class Shape:
    """A piece's shape at the reference turn and size 1, and its renderings."""

    base: np.ndarray
    renders: dict = field(default_factory=dict)  # (k, s) -> mask
    lookup: dict = field(default_factory=dict)  # mask key -> [(k, s)]
    rings: dict = field(default_factory=dict)  # (k, s) -> ring mask

    @classmethod
    def of(cls, mask: np.ndarray) -> "Shape":
        s = _blocky(mask)
        sh = cls(mask[::s, ::s].copy())
        for k in range(4):
            r = np.rot90(sh.base, k)
            for sc in range(1, MAX_SCALE + 1):
                m = np.kron(r, np.ones((sc, sc), dtype=bool))
                sh.renders[(k, sc)] = m
                sh.lookup.setdefault(_key(m), []).append((k, sc))
        return sh, s

    def size(self, k: int, s: int) -> tuple[int, int]:
        h, w = self.renders[(k, s)].shape
        return w, h


def _key(mask: np.ndarray) -> bytes:
    return bytes(str(mask.shape), "ascii") + np.packbits(mask).tobytes()


@dataclass
class Obj:
    color: int
    x1: int
    y1: int
    x2: int
    y2: int
    mask: np.ndarray  # bbox-local

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def rect(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


def _floor(sub: np.ndarray) -> set:
    """A board's floor: the two commonest colours that come in small cells (a
    checkerboard), else its commonest colour."""
    counts = np.bincount(sub.ravel(), minlength=16)
    areas: dict = {}
    for c in fp.segment(sub):
        areas.setdefault(c.color, []).append(c.area)
    small = [c for c in areas if counts[c] >= 0.15 * sub.size and float(np.median(areas[c])) <= CHECKER_CELL]
    if len(small) >= 2:
        small.sort(key=lambda c: -counts[c])
        return set(small[:2])
    return {int(counts.argmax())}


def board_objects(grid: np.ndarray, region: tuple, floor: set | None = None) -> tuple[list[Obj], set]:
    """Single-colour 8-connected objects on a board, and its floor colours."""
    x1, y1, x2, y2 = region
    sub = grid[y1:y2 + 1, x1:x2 + 1]
    if floor is None:
        floor = _floor(sub)
    out = []
    struct = np.ones((3, 3), dtype=bool)
    for colour in np.unique(sub):
        colour = int(colour)
        if colour in floor:
            continue
        m = sub == colour
        if ndimage is not None:
            lab, n = ndimage.label(m, structure=struct)
        else:  # pragma: no cover
            lab, n = fp._label_fallback(m, 8)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            oy, ox = ys.min(), xs.min()
            mask = np.zeros((ys.max() - oy + 1, xs.max() - ox + 1), dtype=bool)
            mask[ys - oy, xs - ox] = True
            out.append(Obj(colour, int(ox + x1), int(oy + y1), int(xs.max() + x1), int(ys.max() + y1), mask))
    return out, floor


def _ring(shape: Shape, k: int, s: int) -> np.ndarray:
    """The pixels within s of the piece rendered (k, s), not under it, in a frame grown by s."""
    key = (k, s)
    if key not in shape.rings:
        m = shape.renders[key]
        h, w = m.shape
        grown = np.zeros((h + 2 * s, w + 2 * s), dtype=bool)
        for dy in range(2 * s + 1):
            for dx in range(2 * s + 1):
                grown[dy:dy + h, dx:dx + w] |= m
        grown[s:s + h, s:s + w] &= ~m
        shape.rings[key] = grown
    return shape.rings[key]


def fit_outline(shape: Shape, outline: Obj) -> tuple | None:
    """Where the piece must stand (x, y, k, s) for the outline to be its ring: every
    outline pixel within s of the piece and none under it, most of the ring drawn."""
    oh, ow = outline.mask.shape
    best = None
    for (k, s), m in shape.renders.items():
        h, w = m.shape
        H, W = h + 2 * s, w + 2 * s
        if ow > W or oh > H:
            continue
        ring = _ring(shape, k, s)
        n_ring = int(ring.sum())
        if 2 * outline.area < n_ring:
            continue
        # the outline inside the grown frame, at every offset it fits
        for oy in range(H - oh + 1):
            for ox in range(W - ow + 1):
                if (ring[oy:oy + oh, ox:ox + ow] | ~outline.mask).all():
                    score = outline.area / n_ring
                    if best is None or score > best[0]:
                        best = (score, (outline.x1 - ox + s, outline.y1 - oy + s, k, s))
    return None if best is None else best[1]


@dataclass
class Board:
    region: tuple
    objects: list
    floor: set


def board_above(grid: np.ndarray, comps: list, panel: Panel) -> Board | None:
    """The board over a panel: inside the panel's ground when it frames one, else the
    region of the colour just above the panel."""
    H, W = grid.shape
    x1, y1, x2, y2 = panel.box
    host = None
    for c in comps:
        if c.color == panel.ground and c.x1 <= x1 and c.x2 >= x2 and c.y1 <= y1 and c.y2 >= y2:
            host = c
            break
    if host is not None and host.y1 < y1 - 6:
        top = y1 - 1
        while top > host.y1 and (grid[top, host.x1 + 1:host.x2] == panel.ground).mean() < 0.9:
            top -= 1
        # the frame's row over the panel: the board is above it
        bottom = top - 1
        while bottom > host.y1 and (grid[bottom, host.x1 + 1:host.x2] == panel.ground).mean() >= 0.9:
            bottom -= 1
        region = (host.x1 + 1, host.y1 + 1, host.x2 - 1, bottom)
    else:
        y = y1 - 1
        while y > 0 and grid[y, panel.cols[0][0].x] == panel.ground:
            y -= 1
        colour = int(grid[y, panel.cols[0][0].x])
        above = next((c for c in comps if c.color == colour and (y, panel.cols[0][0].x) in set(map(tuple, c.pixels.tolist()))), None)
        if above is None:
            return None
        region = (above.x1, above.y1, above.x2, above.y2)
    if region[2] - region[0] < 4 or region[3] - region[1] < 4:
        return None
    objs, floor = board_objects(grid, region)
    return Board(region, objs, floor)


def find_tabs(grid: np.ndarray, window: tuple) -> list[tuple[int, int]]:
    """Centres of a row of equal square frames (at least two) starting in the window
    (x1, y1, x2, y2), left to right."""
    H, W = grid.shape
    found = []
    for w in range(7, 12):
        for y in range(window[1], min(window[3], H - w) + 1):
            for x in range(window[0], min(window[2], W - w) + 1):
                top = grid[y, x:x + w]
                b = top[0]
                if (top != b).any() or (grid[y + w - 1, x:x + w] != b).any():
                    continue
                if (grid[y:y + w, x] != b).any() or (grid[y:y + w, x + w - 1] != b).any():
                    continue
                inner = grid[y + 1:y + w - 1, x + 1:x + w - 1]
                if (inner == b).all() or (inner[0] == b).all() or (inner[:, 0] == b).all():
                    continue
                found.append((y, w, x))
    rows: dict = {}
    for y, w, x in found:
        rows.setdefault((y, w), []).append(x)
    best: list = []
    for (y, w), xs in rows.items():
        xs = sorted(xs)
        keep = [xs[0]]
        for v in xs[1:]:
            if v >= keep[-1] + w:
                keep.append(v)
        if len(keep) >= 2 and len(keep) > len(best):
            best = [(v + w // 2, y + w // 2) for v in keep]
    return best


# --- the planner -------------------------------------------------------------------

State = tuple  # (x, y, k, s, colour)


def apply(state: State, effect: tuple) -> State:
    x, y, k, s, c = state
    dx, dy, dk, ds, nc = effect
    return (x + dx, y + dy, (k + dk) % 4, max(1, s + ds), c if nc is None else nc)


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


class World:
    """What a board allows: the region, obstacles (a pixel map), marks, the piece's shape."""

    def __init__(self, region: tuple, obstacles: np.ndarray, marks: list, shape: Shape,
                 hazards: list | None = None) -> None:
        self.region = region
        self.obstacles = obstacles
        self.marks = marks
        self.shape = shape
        self.hazards = hazards or []  # per frame of a run, the pixels that are lit then

    def rect(self, st: State) -> tuple:
        w, h = self.shape.size(st[2], st[3])
        return (st[0], st[1], st[0] + w - 1, st[1] + h - 1)

    def ok(self, st: State) -> bool:
        if st[3] > MAX_SCALE:
            return False
        r = self.rect(st)
        x1, y1, x2, y2 = self.region
        if r[0] < x1 or r[1] < y1 or r[2] > x2 or r[3] > y2:
            return False
        return not self.obstacles[r[1]:r[3] + 1, r[0]:r[2] + 1].any()

    def safe(self, st: State, frame: int) -> bool:
        """Whether the piece stands clear of what is lit in that frame of a run."""
        if frame >= len(self.hazards):
            return True
        r = self.rect(st)
        return not self.hazards[frame][r[1]:r[3] + 1, r[0]:r[2] + 1].any()

    def on_mark(self, st: State) -> bool:
        r = self.rect(st)
        return any(r == m for m in self.marks)


def plan_run(world: World, start: State, columns: list, options: list) -> dict:
    """Every end state one run can reach: end -> (cost, values). options[i] is a list of
    (effect, cost, value) for column i."""
    layer = {start: (0, ())}
    for i in range(len(columns)):
        nxt: dict = {}
        for st, (cost, vals) in layer.items():
            for effect, c, v in options[i]:
                ns = apply(st, effect)
                if ns != st and not world.ok(ns):
                    continue
                if not world.safe(ns, i + 1):
                    continue
                nc = cost + c
                if ns not in nxt or nc < nxt[ns][0]:
                    nxt[ns] = (nc, vals + (v,))
        layer = nxt
    return layer


# --- the solver ----------------------------------------------------------------------


def _pixels_of(grid: np.ndarray, shape: Shape, st: State) -> bool:
    """Whether the piece in state st is drawn there (every pixel of it in its colour)."""
    m = shape.renders.get((st[2], st[3]))
    if m is None:
        return False
    h, w = m.shape
    x, y = st[0], st[1]
    if x < 0 or y < 0 or y + h > grid.shape[0] or x + w > grid.shape[1]:
        return False
    return bool((grid[y:y + h, x:x + w][m] == st[4]).all())


def _inside(shape: Shape, o: Obj) -> list:
    """States in which the piece is drawn wholly within an object's pixels."""
    out = []
    oh, ow = o.mask.shape
    for (k, s), m in shape.renders.items():
        h, w = m.shape
        if h > oh or w > ow:
            continue
        for y in range(oh - h + 1):
            for x in range(ow - w + 1):
                if (o.mask[y:y + h, x:x + w] | ~m).all():
                    out.append((o.x1 + x, o.y1 + y, k, s, o.color))
    return out


class BitSolver:
    name = "bits"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.effects: dict = {}  # value -> (dx, dy, dk, ds, colour or None)
        self.on_off: tuple | None = None  # (on colour, off colour)
        self.marks_work = True
        self.shape_prior: Shape | None = None  # the piece's shape on the last level
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.fails = 0
        self.actions = 0
        self.scene: dict | None = None
        self.queue: list = []
        self.expect: tuple | None = None
        self.tabs_seen: set = set()
        self.probed = False
        self.start: State | None = None  # where the piece should be between runs
        self.blinking: set = set()  # colours on the board that come and go during a run
        self.hazards: list = []  # per frame of a run, where they are lit

    # --- reading -----------------------------------------------------------------

    def _bit_colours(self, panels: list[Panel]) -> None:
        counts = Counter(g.color for p in panels for col in p.cols for g in col)
        if len(counts) >= 2 and self.on_off is None:
            (off, _), (on, _) = counts.most_common(2)
            self.on_off = (on, off)

    def _value(self, pattern: tuple) -> int | None:
        if self.on_off is None:
            return 0 if len(set(pattern)) == 1 else None  # all one colour: nothing on
        on, off = self.on_off
        v = 0
        for i, c in enumerate(pattern):
            if c == on:
                v |= 1 << i
            elif c != off:
                return None
        return v

    def _read(self, grid: np.ndarray) -> dict | None:
        """The whole scene, from a level's first frame."""
        comps = fp.segment(grid)
        panels = find_panels(grid, comps)
        if not panels:
            return None
        boards = []
        for p in panels:
            b = board_above(grid, comps, p)
            if b is not None:
                boards.append((p, b))
        mine = None
        for p, b in boards:
            mine = self._piece_and_target(b)
            if mine:
                break
        if mine is None:
            return None
        self._bit_colours([q for q, _ in boards])
        piece, shape, start, o, fit = mine
        demo = next(((dp, db) for dp, db in boards if dp is not p), None)
        avoid = [b.region, p.box] + ([demo[0].box, demo[1].region] if demo else [])
        tabs = []
        if demo is not None:
            ground = next((c for c in comps if c.color == demo[0].ground and c.x1 <= demo[0].box[0]
                           and c.x2 >= demo[0].box[2] and c.y1 <= demo[0].box[1] and c.y2 >= demo[0].box[3]), None)
            x1, x2 = (ground.x1, ground.x2) if ground is not None else (demo[1].region[0], demo[1].region[2])
            tabs = find_tabs(grid, (max(x1 - 1, 0), demo[0].box[3] + 2, x2, grid.shape[0] - 1))
        # the button: the solid object off the boards and tabs nearest the panel
        button, best = None, None
        for c in comps:
            if c.width < 3 or c.height < 3 or c.area < 12 or c.area > 400 or c.color == p.ground:
                continue
            if demo is not None and c.color == demo[0].ground:
                continue
            if any(c.x1 <= t[0] <= c.x2 and c.y1 <= t[1] <= c.y2 for t in tabs):
                continue
            cx, cy = c.click_point()
            if any(a[0] <= cx <= a[2] and a[1] <= cy <= a[3] for a in avoid):
                continue
            d = abs(cx - (p.box[0] + p.box[2]) / 2) + abs(cy - p.box[3])
            if best is None or d < best:
                best, button = d, (cx, cy)
        if button is None:
            return None
        tx, ty, tk, ts = fit
        others = [m for m in b.objects if m is not piece and m is not o]
        blocked = np.zeros(grid.shape, dtype=bool)
        for m in others:
            if m.color not in (piece.color, o.color):
                blocked[m.y1:m.y2 + 1, m.x1:m.x2 + 1] |= m.mask
        return dict(
            panel=p, region=b.region, floor=b.floor, shape=shape, start=start,
            target=(tx, ty, tk, ts, o.color), button=button, tabs=tabs, demo=demo, demo_shape=None,
            marks=[m.rect() for m in others if m.color in (piece.color, o.color)],
            obstacles=blocked,
        )

    def _piece_and_target(self, b: Board) -> tuple | None:
        """The piece and its outline on a board: (piece object, shape, state, outline
        object, target fit). The last level's piece shape is tried first, then each
        object's own, then the last shape inside an object (a piece on a mark of its
        colour merges with it)."""
        objs = [o for o in b.objects if o.area >= 4]
        prior = self.shape_prior
        cands = []
        if prior is not None:
            for o in objs:
                for k, sc in prior.lookup.get(_key(o.mask), []):
                    cands.append((o, prior, (o.x1, o.y1, k, sc, o.color)))
        for o in objs:
            if not o.mask.all():  # a solid rectangle shows no turn: no piece
                shape, s0 = Shape.of(o.mask)
                cands.append((o, shape, (o.x1, o.y1, 0, s0, o.color)))
        if prior is not None:
            for o in objs:
                if not o.mask.all():
                    cands.extend((o, prior, st) for st in _inside(prior, o))
        fits: dict = {}
        for piece, shape, st in cands:
            for t in objs:
                if t is piece:
                    continue
                key = (id(shape), id(t))
                if key not in fits:
                    fits[key] = fit_outline(shape, t)
                if fits[key] is not None:
                    return piece, shape, st, t, fits[key]
        return None

    def _panel_now(self, grid: np.ndarray) -> Panel | None:
        sc = self.scene
        for p in find_panels(grid, fp.segment(grid)):
            if p.ground == sc["panel"].ground and p.box == sc["panel"].box:
                return p
        return None

    def _piece_in(self, grid: np.ndarray, region: tuple, floor: set, shape: Shape, near: State | None = None) -> State | None:
        objs, _ = board_objects(grid, region, floor)
        found = []
        for o in objs:
            for k, s in shape.lookup.get(_key(o.mask), []):
                found.append((o.x1, o.y1, k, s, o.color))
        if not found and near is not None:
            if _pixels_of(grid, shape, near):
                return near  # still where it was, on something of its colour
            # merged with something of its colour: drawn inside an object, nearest first
            for o in objs:
                if o.x2 < near[0] - 16 or o.x1 > near[0] + 16 or o.y2 < near[1] - 16 or o.y1 > near[1] + 16:
                    continue
                found.extend(_inside(shape, o))
        if not found:
            return None
        if near is not None:
            found.sort(key=lambda st: (abs(st[0] - near[0]) + abs(st[1] - near[1]) + (st[4] != near[4]),
                                       (st[2] != near[2]) + (st[3] != near[3])))
        return found[0]

    # --- learning from a run -----------------------------------------------------

    def _learn(self, frames: list, region: tuple, floor: set, shape: Shape, values: list, first: State | None) -> list:
        """The piece's states over a run's frames; what each column did is recorded."""
        states = []
        prev = first
        for f in frames:
            st = self._piece_in(np.asarray(f), region, floor, shape, prev)
            states.append(st)
            if st is not None:
                prev = st
        for i, v in enumerate(values):
            if i + 1 >= len(states):
                break
            a, b = states[i], states[i + 1]
            if a is None or b is None or v is None:
                continue
            effect = (b[0] - a[0], b[1] - a[1], (b[2] - a[2]) % 4, b[3] - a[3], b[4] if b[4] != a[4] else None)
            old = self.effects.get(v)
            if effect == (0, 0, 0, 0, None):
                # nothing seen: what it does may have been stopped; keep what is known
                if old is None:
                    self.effects[v] = effect
                continue
            # an instruction does one thing: a move, a turn, a resize or a recolour
            kinds = (effect[0] != 0 or effect[1] != 0) + (effect[2] != 0) + (effect[3] != 0) + (effect[4] is not None)
            if kinds != 1:
                self.stats["odd_change"] += 1
                continue
            if old is None or old == (0, 0, 0, 0, None):
                self.effects[v] = effect
                self.stats["learned"] += 1
            elif old != effect:
                self.stats["conflicts"] += 1
        return states

    def _watch_blinking(self, frames: list, states: list) -> None:
        """Colours on the board that come and go during a run, other than the piece's,
        the target's and the floor's, are hazards; where they are lit in each frame of
        the run is kept (a run's rhythm is taken to be every run's)."""
        sc = self.scene
        x1, y1, x2, y2 = sc["region"]
        keep = set(sc["floor"]) | {sc["start"][4], sc["target"][4]} | {st[4] for st in states if st is not None}
        subs = [np.asarray(f)[y1:y2 + 1, x1:x2 + 1] for f in frames]
        for a, b in zip(subs, subs[1:]):
            changed = a != b
            for c in set(np.unique(a[changed]).tolist()) | set(np.unique(b[changed]).tolist()):
                if int(c) not in keep and int(c) not in self.blinking:
                    self.blinking.add(int(c))
                    self.stats["hazard_colours"] += 1
        if not self.blinking:
            return
        H, W = np.asarray(frames[0]).shape
        maps = []
        for sub in subs:
            m = np.zeros((H, W), dtype=bool)
            m[y1:y2 + 1, x1:x2 + 1] = np.isin(sub, list(self.blinking))
            maps.append(m)
        self.hazards = maps

    # --- planning ----------------------------------------------------------------

    def _options(self, patterns: list[tuple]) -> list:
        """Per column, the cheapest way to each known effect: (effect, flips, value)."""
        on = self.on_off[0]
        out = []
        for pat in patterns:
            best: dict = {}
            for v, eff in self.effects.items():
                if v >= 1 << len(pat):
                    continue
                flips = sum(1 for i, c in enumerate(pat) if bool(v >> i & 1) != (c == on))
                if eff not in best or flips < best[eff][0]:
                    best[eff] = (flips, v)
            out.append([(eff, f, v) for eff, (f, v) in best.items()])
        return out

    def _plan(self, start: State, patterns: list[tuple]) -> list | None:
        """The cheapest chain of runs to the target: [(values, expected end)]."""
        if self.on_off is None or not self.effects:
            return None
        sc = self.scene
        world = World(sc["region"], sc["obstacles"], sc["marks"] if self.marks_work else [], sc["shape"], self.hazards)
        target = sc["target"]
        on, off = self.on_off
        heap = [(0, 0, start, tuple(patterns), ())]
        seen: dict = {}
        counter = 0
        while heap:
            cost, _, st, pats, runs = heapq.heappop(heap)
            if len(runs) >= MAX_RUNS:
                continue
            ends = plan_run(world, st, list(pats), self._options(list(pats)))
            if target in ends:
                return list(runs) + [(ends[target][1], target)]
            for end, (c, vals) in ends.items():
                if end == st or not world.on_mark(end):
                    continue
                total = cost + c + 1
                if end in seen and seen[end] <= total:
                    continue
                seen[end] = total
                newpats = tuple(tuple(on if v >> i & 1 else off for i in range(len(p))) for v, p in zip(vals, pats))
                counter += 1
                heapq.heappush(heap, (total, counter, end, newpats, runs + ((vals, end),)))
        return None

    # --- watching ----------------------------------------------------------------

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        if action is None or action is GameAction.RESET:
            self.queue, self.expect = [], None
            return
        if self.expect is None or self.scene is None:
            return
        kind, info = self.expect
        self.expect = None
        sc = self.scene
        frames = list(obs.frame)
        if kind == "flip":
            now = int(np.asarray(frames[-1])[info.y, info.x])
            if now == info.color:
                self.fails += 1
                self.stats["flip_failed"] += 1
                self.queue = []
            elif self.on_off is None:
                self.on_off = (now, info.color)
            return
        if kind == "tab":
            demo = sc["demo"]
            if len(frames) < 3:
                return
            first = np.asarray(frames[0])
            panels = [p for p in find_panels(first, fp.segment(first)) if p.ground == demo[0].ground]
            if not panels:
                return
            values = [self._value(pat) for pat in panels[0].patterns()]
            region, floor = demo[1].region, demo[1].floor
            if sc["demo_shape"] is None:
                objs, _ = board_objects(first, region, floor)
                objs = [o for o in objs if o.area >= 4]
                if not objs:
                    return
                sc["demo_shape"], _ = Shape.of(max(objs, key=lambda o: o.area).mask)
            self._learn(frames, region, floor, sc["demo_shape"], values, None)
            self.stats["demos"] += 1
            return
        if kind == "run":
            values, expect_end = info
            states = self._learn(frames, sc["region"], sc["floor"], sc["shape"], values, self.start)
            self.stats["runs"] += 1
            self._watch_blinking(frames, states)
            n = len(values)
            end = states[n] if n < len(states) else None
            last = next((s for s in reversed(states) if s is not None), None)
            if expect_end is None:  # a probe
                if last is not None:
                    self.start = last
                return
            if end != expect_end:
                self.fails += 1
                self.stats["run_missed"] += 1
                if last is not None:
                    self.start = last
                return
            if last is not None and last[:2] != expect_end[:2]:
                # it ended on a mark and the start did not move there
                self.marks_work = False
                self.stats["marks_refuted"] += 1
            if last is not None:
                self.start = last

    # --- acting ------------------------------------------------------------------

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if 6 not in obs.available_actions:
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if not self.queue:
            if self.scene is None:
                if self.read:
                    return None
                self.read = True
                sc = self._read(grid)
                if sc is None:
                    self.gave_up = True  # not this family's screen (looked at once a level)
                    return None
                self.scene = sc
                self.start = sc["start"]
                self.shape_prior = sc["shape"]
            sc = self.scene
            panel = self._panel_now(grid)
            start = self._piece_in(grid, sc["region"], sc["floor"], sc["shape"], self.start)
            if panel is None or start is None:
                self.stats["lost_sight"] += 1
                self._give_up()
                return None
            self.start = start
            sc["panel"] = panel
            plan = self._plan(start, panel.patterns())
            if plan is None:
                tab = next((t for t in sc["tabs"] if t not in self.tabs_seen), None)
                if tab is not None:
                    self.tabs_seen.add(tab)
                    self.expect = ("tab", tab)
                    return self._click(tab)
                if not self.probed:
                    # run what the panel holds, to see what its values do
                    self.probed = True
                    self.expect = ("run", ([self._value(p) for p in panel.patterns()], None))
                    return self._click(sc["button"])
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            values, end = plan[0]
            on, off = self.on_off
            for col, v in zip(panel.cols, values):
                for i, g in enumerate(col):
                    if g.color != (on if v >> i & 1 else off):
                        self.queue.append(("flip", g))
            self.queue.append(("run", (list(values), end)))
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        kind, info = self.queue.pop(0)
        self.expect = (kind, info)
        if kind == "flip":
            return self._click((info.x, info.y))
        return self._click(self.scene["button"])

    def _click(self, at: tuple[int, int]) -> Action:
        self.actions += 1
        self.stats["clicks"] += 1
        return GameAction.ACTION6, {"x": int(at[0]), "y": int(at[1])}
