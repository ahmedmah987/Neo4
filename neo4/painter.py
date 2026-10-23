"""The copy family: paint a canvas until it matches a reference shown beside it.

Some puzzles show a small picture and, elsewhere, a blank canvas of the same size,
and are solved when the canvas reproduces the picture. The canvas is painted with
tools: a cursor that moves between a few places around it, a key that paints the
cursor's region of the canvas with the selected colour, stamps to click next to
the cursor, a palette of colours to click. This solver learns those tools from
what it sees and plans the painting.

From the frames alone:

* the *reference* and the *canvas* are two isolated rectangles of the same size
  (no background inside, background all round); the canvas is the one an action
  paints, i.e. turns a set of its pixels into one colour;
* the *cursor* is the largest non-background shape in a margin round the canvas
  (the canvas left out); actions that change it without painting are cursor
  moves, learned as a graph over cursor states;
* a *brush* is an action at a cursor state that paints. A use shows the pixels it
  changed (inside its region) and the pixels it left although they had another
  colour (outside); pixels that already had the colour stay unknown. The region
  is completed with simple shapes, axis-aligned rectangles and the corner
  triangles of a square canvas: what every shape that fits the evidence covers
  is sure, what only some cover is uncertain;
* the *palette* is learned from clicks that change the frame but no canvas pixel
  and no cursor state: the clicked pixel's colour becomes the colour of the next
  brush.

It plans by peeling the reference in reverse: the stroke painted last is a brush
whose region, among the pixels not yet explained, is all one colour in the
reference; those pixels are explained, and the search goes on until the pixels
left already match the canvas. Pixels that differed from the reference when a
level was completed are not checked by the game and are left out. When no exact
plan exists, a plan that leaves a few pixels wrong is tried (the game may not
check them); if the canvas then matches all but those pixels and the level goes
on, they are checked. Then it walks the cursor, selects colours and paints,
re-planning after every action. On the first level it surveys first: every
cursor place it can reach, every brush tried once, because later levels need
brushes the first does not. What it lacks it explores: places and brushes not
tried, brushes whose region is still uncertain, clickable things next to the
cursor.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from functools import lru_cache

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

MIN_SIDE = 5  # rectangles smaller than this are not pictures
MAX_LEVEL_ACTIONS = 300  # actions the solver may spend on one level
UNPROVEN_ACTIONS = 120  # ... in a game it has not completed a level of
MOVES = (1, 2, 3, 4)  # keyboard actions that may move the cursor
MARGIN = 14  # the cursor is read within this many pixels of the canvas
PAINT_KEYS = (5,)  # keyboard actions that may paint
MAX_STROKES = 64  # strokes in one plan
CONFLICT_COST = 4  # a pixel an approximate plan leaves wrong, in pixels painted right
STAMP_AREA = 64  # clickable things next to the cursor are at most this large


def windows(grid: np.ndarray, bg: int, mask: np.ndarray | None) -> list[tuple[int, int, int, int]]:
    """Isolated filled rectangles (x, y, w, h): connected regions without a
    background pixel whose bounding box they fill entirely."""
    solid = grid != bg
    if mask is not None:
        solid &= ~mask
    out = []
    for c in fp.segment(solid.astype(np.int8)):
        if c.color != 1:
            continue
        w, h = c.width, c.height
        if w >= MIN_SIDE and h >= MIN_SIDE and c.area == w * h:
            out.append((c.x1, c.y1, w, h))
    return out


@lru_cache(maxsize=None)
def triangles(h: int, w: int) -> tuple[np.ndarray, ...]:
    """The corner triangles of a square canvas, with and without their diagonal."""
    if h != w:
        return ()
    i, j = np.indices((h, w))
    n = h - 1
    return (j <= n - i, j < n - i, j >= i, j > i, j >= n - i, j > n - i, j <= i, j < i)


def complete(inside: np.ndarray, outside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sure, uncertain) pixels of a brush's region from what its uses showed.

    Hypotheses are the rectangles and corner triangles that contain every pixel
    seen inside and none seen outside; the pixels all of them cover are sure, the
    pixels only some cover uncertain. When no hypothesis fits, the region is
    what was seen inside, and every pixel not seen either way is uncertain."""
    if not inside.any():
        return inside.copy(), ~outside
    h, w = inside.shape
    inter = np.ones_like(inside)
    union = np.zeros_like(inside)
    fits = False
    ys, xs = np.nonzero(inside)
    by0, by1, bx0, bx1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if not outside[by0:by1 + 1, bx0:bx1 + 1].any():
        # every rectangle that fits contains the bounding box; the union of those
        # that fit, row span by row span, extended sideways as far as allowed
        fits = True
        inter[:] = False
        inter[by0:by1 + 1, bx0:bx1 + 1] = True
        for y0 in range(by0, -1, -1):
            if outside[y0, bx0:bx1 + 1].any():
                break
            for y1 in range(by1, h):
                if outside[y1, bx0:bx1 + 1].any():
                    break
                bad = outside[y0:y1 + 1].any(axis=0)
                xl, xr = bx0, bx1
                while xl > 0 and not bad[xl - 1]:
                    xl -= 1
                while xr < w - 1 and not bad[xr + 1]:
                    xr += 1
                union[y0:y1 + 1, xl:xr + 1] = True
    for t in triangles(h, w):
        if (inside & ~t).any() or (t & outside).any():
            continue
        fits = True
        inter &= t
        union |= t
    if not fits:
        return inside.copy(), ~(inside | outside)
    sure = inside | inter
    return sure, union & ~sure


class PainterSolver:
    name = "painter"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.confirmed = False
        self.surveyed = False
        self.strikes = 0
        self.stats: Counter = Counter()
        self.ignore: dict[tuple, np.ndarray] = {}  # canvas shape -> pixels the game does not check
        self.checked: dict[tuple, np.ndarray] = {}  # canvas shape -> pixels shown to be checked
        # What the tools do carries over between levels: cursor states are read
        # round the canvas, relative to it.
        self.edges: dict[str, dict[tuple, str]] = {}  # cursor state -> op -> cursor state
        self.tried: dict[str, set] = {}  # cursor state -> ops tried there
        self.brushes: dict[tuple, list[np.ndarray]] = {}  # (cursor state, op) -> [inside, outside]
        self._regions: dict[tuple, tuple] = {}
        self.last_canvas: tuple | None = None
        self.level = -1
        self.in_control = False
        self._new_level(None)

    # -- per level -------------------------------------------------------------

    def _new_level(self, grid: np.ndarray | None) -> None:
        self.grid = grid
        self.mask = fp.status_bar_mask(fp.segment(grid)) if grid is not None else None
        self.bg = int(np.bincount(grid.ravel()).argmax()) if grid is not None else 0
        self.pairs: list[tuple[tuple, tuple]] = []
        if grid is not None:
            ws = windows(grid, self.bg, self.mask)
            for i, a in enumerate(ws):
                for b in ws[i + 1:]:
                    if a[2:] == b[2:] and not np.array_equal(self._cut(grid, a), self._cut(grid, b)):
                        self.pairs.append((a, b))
        self.canvas: tuple | None = None
        self.reference: tuple | None = None
        if self.confirmed and self.last_canvas is not None:
            # a game this solver has painted before: the canvas is where it was
            for a, b in self.pairs:
                if self.last_canvas in (a, b):
                    self.canvas = self.last_canvas
                    self.reference = b if a == self.last_canvas else a
        self.swatches: dict[int, tuple[int, int]] = {}  # colour -> click point
        self.dead: set = set()  # click points that changed nothing this level
        self.blobs_seen: dict[str, list] = {}  # cursor place -> things next to the cursor seen there this level
        self.new_stamp = False  # a click painted for the first time this level
        self.unverified: tuple | None = None  # (colour, point) assumed selected by a click that changed nothing
        self.colour: int | None = None
        self.gave_up = False
        self.level_actions = 0
        self.deaths = 0

    @staticmethod
    def _cut(grid: np.ndarray, w: tuple) -> np.ndarray:
        x, y, ww, hh = w
        return grid[y:y + hh, x:x + ww]

    def _cursor(self, grid: np.ndarray):
        """The largest non-background shape in a margin round the canvas (the
        canvas left out), with the margin's corner."""
        x, y, w, h = self.canvas
        x0, y0 = max(x - MARGIN, 0), max(y - MARGIN, 0)
        x1, y1 = min(x + w + MARGIN, grid.shape[1]), min(y + h + MARGIN, grid.shape[0])
        m = grid[y0:y1, x0:x1] != self.bg
        m[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
        if self.mask is not None:
            m &= ~self.mask[y0:y1, x0:x1]
        comps = [c for c in fp.segment(m.astype(np.int8)) if c.color == 1]
        if not comps:
            return None
        return max(comps, key=lambda c: c.area), x0, y0

    def _key(self, grid: np.ndarray) -> str:
        """The cursor state: the cursor's outline relative to the canvas, so that
        it reads the same at every level where the canvas sits alike."""
        cur = self._cursor(grid)
        if cur is None:
            return "none"
        c, x0, y0 = cur
        pts = c.pixels + np.array([y0 - self.canvas[1], x0 - self.canvas[0]])
        return hashlib.blake2b(pts.astype(np.int16).tobytes(), digest_size=8).hexdigest()

    # -- observing --------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled:
            return
        if obs.state == GameState.GAME_OVER and self.in_control:
            self.deaths += 1
        if obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        grid = np.asarray(obs.frame[-1])
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0:
                if len(obs.frame) > 1:
                    # the frame before the new level's own is the finished level's last
                    self._learn_ignored(np.asarray(obs.frame[-2]))
                if self.in_control:
                    self.confirmed = True
                    self.stats["levels"] += 1
                    self.last_canvas = self.canvas
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level(grid)
            return
        prev = self.grid
        self.grid = grid
        if self.canvas is not None:
            self.blobs_seen.setdefault(self._key(grid), self._blobs(grid))
        if prev is None or action is None or action is GameAction.RESET or not self.pairs:
            return
        op = (action.value, (int(data["x"]), int(data["y"])) if data else None)
        painted = self._painted(prev, grid)
        if painted is not None:
            win, changed, colour = painted
            if self.canvas is None:
                self.canvas = win
                self.reference = next(b if a == win else a for a, b in self.pairs if win in (a, b))
                self.stats["recognised"] += 1
            key = self._key(prev)
            before = self._cut(prev, self.canvas)
            if op[1] is not None and (key, op) not in self.brushes:
                self.new_stamp = True  # a new kind of tool: look for more at every place
            b = self.brushes.setdefault((key, op), [np.zeros_like(changed), np.zeros_like(changed)])
            if b[0].shape == changed.shape:
                b[0] |= changed
                b[1] |= (before != colour) & ~changed
            self.tried.setdefault(key, set()).add(op)
            if self.unverified is not None and self.unverified[0] != colour:
                c, pt = self.unverified
                self.dead.add(pt)
                if self.swatches.get(c) == pt:
                    del self.swatches[c]
            self.unverified = None
            self.colour = colour
            return
        if self.canvas is None:
            return
        key_prev, key_new = self._key(prev), self._key(grid)
        self.tried.setdefault(key_prev, set()).add(op)
        if op[1] is None:
            if key_new != key_prev:
                self.edges.setdefault(key_prev, {})[op] = key_new
            elif action.value in PAINT_KEYS and self.colour is not None:
                # a paint key that changed nothing: its region, if any, holds only
                # pixels that already had the colour
                before = self._cut(prev, self.canvas)
                b = self.brushes.setdefault((key_prev, op), [np.zeros(before.shape, bool), np.zeros(before.shape, bool)])
                if b[1].shape == before.shape:
                    b[1] |= before != self.colour
            return
        # A click that painted nothing: if the frame changed (the status bar left
        # out), a palette choice; the clicked pixel's colour is the new colour.
        x, y = op[1]
        changed = prev != grid
        if self.mask is not None:
            changed &= ~self.mask
        c = int(prev[y, x])
        swatch = c != self.bg and self._outside(x, y) and c in set(np.unique(self._cut(prev, self.reference)).tolist())
        if not changed.any():
            if swatch and self.colour is None and (x, y) not in self.dead:
                # perhaps the colour already selected: the next stroke tells
                self.unverified = (c, (x, y))
                self.swatches[c] = (x, y)
                self.colour = c
            else:
                self.dead.add((x, y))
            return
        if swatch:
            self.swatches[c] = (x, y)
            self.colour = c
            self.stats["palette_clicks"] += 1

    def _painted(self, prev: np.ndarray, grid: np.ndarray):
        """(window, changed pixels, colour) when the transition turned pixels of
        one of the paired windows into a single colour; else None."""
        cands = [self.canvas] if self.canvas else [w for pair in self.pairs for w in pair]
        for w in cands:
            a, b = self._cut(prev, w), self._cut(grid, w)
            diff = a != b
            if not diff.any():
                continue
            cols = np.unique(b[diff])
            if len(cols) == 1:
                return w, diff, int(cols[0])
        return None

    def _learn_ignored(self, final: np.ndarray) -> None:
        """At a level-up the frames before the new level's still show the finished
        canvas: the pixels that differ from the reference are not checked."""
        if self.canvas is None or self.reference is None:
            return
        a, b = self._cut(final, self.canvas), self._cut(final, self.reference)
        if a.shape != b.shape:
            return
        diff = a != b
        if a.shape in self.ignore:
            self.ignore[a.shape] |= diff
        else:
            self.ignore[a.shape] = diff
        self.stats["ignore_learned"] += int(diff.sum())

    # -- planning ---------------------------------------------------------------

    def _region(self, bkey: tuple) -> tuple[np.ndarray, np.ndarray]:
        inside, outside = self.brushes[bkey]
        n = (int(inside.sum()), int(outside.sum()))
        hit = self._regions.get(bkey)
        if hit is None or hit[0] != n:
            hit = (n, *complete(inside, outside))
            self._regions[bkey] = hit
        return hit[1], hit[2]

    def _decompose(self, target: np.ndarray, canvas: np.ndarray, colours: set[int], budget: int = 0):
        """(strokes, pixels left wrong): strokes (brush, colour) in painting order
        that turn ``canvas`` into ``target`` on the checked pixels, but for at most
        ``budget`` pixels left wrong; None when the known brushes cannot."""
        shape = target.shape
        todo = ~self.ignore.get(shape, np.zeros(shape, bool))
        checked = self.checked.get(shape, np.zeros(shape, bool))
        regions = [(k, *self._region(k)) for k, (ins, _) in self.brushes.items() if ins.shape == shape]
        wrong = np.zeros(shape, bool)
        strokes: list[tuple] = []
        for _ in range(MAX_STROKES):
            left = todo & (canvas != target)
            if not left.any():
                return strokes[::-1], wrong
            if budget and int(left.sum()) + int(wrong.sum()) <= budget and not (left & checked).any():
                return strokes[::-1], wrong | left
            best = None
            for k, sure, maybe in regions:
                part = sure & todo
                n = int(part.sum())
                if n == 0:
                    continue
                vals = Counter(target[part].tolist())
                c = next((v for v, _ in vals.most_common() if v in colours), None)
                if c is None:
                    continue
                bad = (part | (maybe & todo)) & (target != c)
                nb = int(bad.sum())
                if nb > budget - int(wrong.sum()) or (bad & checked).any():
                    continue
                score = (n - CONFLICT_COST * nb, int((part & (canvas != target)).sum()))
                if best is None or score > best[0]:
                    best = (score, k, c, part, bad & part)
            if best is None:
                return None
            _, k, c, part, bad = best
            strokes.append((k, c))
            wrong |= bad
            todo &= ~part
        return None

    def _prune(self, strokes: list[tuple], target: np.ndarray, canvas: np.ndarray) -> list[tuple]:
        """Drop strokes whose absence leaves no checked pixel newly wrong."""
        todo = ~self.ignore.get(target.shape, np.zeros(target.shape, bool))

        def wrong(ss):
            out = canvas.copy()
            for k, c in ss:
                out[self._region(k)[0]] = c
            return todo & (out != target)

        base = wrong(strokes)
        i = 0
        while i < len(strokes):
            rest = strokes[:i] + strokes[i + 1:]
            if not (wrong(rest) & ~base).any():
                strokes = rest
            else:
                i += 1
        return strokes

    def _route(self, src: str, dst: str) -> list[tuple] | None:
        """Cursor moves from one state to another over the learned graph."""
        if src == dst:
            return []
        prev = {src: None}
        q = deque([src])
        while q:
            k = q.popleft()
            for op, nk in self.edges.get(k, {}).items():
                if nk not in prev:
                    prev[nk] = (k, op)
                    if nk == dst:
                        path = []
                        while prev[nk] is not None:
                            k2, o = prev[nk]
                            path.append(o)
                            nk = k2
                        return path[::-1]
                    q.append(nk)
        return None

    def _make_plan(self, approximate: bool = False, again: bool = True) -> list[tuple] | None:
        """The next action of a plan that paints the reference (with
        ``approximate``, one that may leave a few pixels wrong); [] when the
        canvas already matches it; None when no plan is known."""
        target = self._cut(self.grid, self.reference)
        canvas = self._cut(self.grid, self.canvas)
        colours = self._palette() | ({self.colour} if self.colour is not None else set())
        plan = self._decompose(target, canvas, colours)
        if plan is None and approximate:
            plan = self._decompose(target, canvas, colours, budget=sum(target.shape))
        if plan is None:
            return None
        strokes, wrong = plan
        if not strokes:
            if wrong.any() and again:
                # as close as planned and the level goes on: those pixels count
                self.checked[target.shape] = self.checked.get(target.shape, np.zeros(target.shape, bool)) | wrong
                self.stats["checked_learned"] += int(wrong.sum())
                return self._make_plan(approximate, again=False)
            return []
        strokes = self._prune(strokes, target, canvas)
        key = self._key(self.grid)
        colour = self.colour
        out: list[tuple] = []
        for (bkey, op), c in strokes:
            if c != colour:
                pt = self.swatches.get(c) or self._find_colour(c)
                if pt is None:
                    return None
                out.append((6, pt))
                colour = c
            route = self._route(key, bkey)
            if route is None:
                return None
            out.extend(route)
            out.append(op)
            key = bkey
        return out[:1] if out else []  # one step at a time: re-plan after each

    # -- exploring --------------------------------------------------------------

    def _survey(self, obs: FrameDataRaw) -> tuple | None:
        """The next step towards a cursor place or a brush not tried yet."""
        key = self._key(self.grid)
        avail = set(obs.available_actions)
        tried = self.tried.get(key, set())
        for a in PAINT_KEYS + MOVES:
            if a in avail and (a, None) not in tried:
                if a in PAINT_KEYS and 6 in avail:
                    # paint in the colour the canvas has least of, so that the
                    # brush's region shows
                    pt = self._rare_colour()
                    if pt is not None:
                        return (6, pt)
                return (a, None)
        places = set(self.tried) | {k for e in self.edges.values() for k in e.values()}
        for k in sorted(places):
            if k == key:
                continue
            if any(a in avail and (a, None) not in self.tried.get(k, set()) for a in PAINT_KEYS + MOVES):
                route = self._route(key, k)
                if route:
                    return route[0]
        return None

    def _explore(self, obs: FrameDataRaw) -> tuple | None:
        step = self._survey(obs)
        if step is not None:
            return step
        avail = set(obs.available_actions)
        key = self._key(self.grid)
        # a brush whose region is still uncertain, painted where that shows most
        canvas = self._cut(self.grid, self.canvas)
        colours = self._palette() | ({self.colour} if self.colour is not None else set())
        best = None
        for bkey in self.brushes:
            sure, maybe = self._region(bkey)
            if sure.shape != canvas.shape or not maybe.any():
                continue
            for c in colours:
                n = int((maybe & (canvas != c)).sum())
                if n and (best is None or n > best[0]):
                    best = (n, bkey, c)
        if best is not None:
            _, (bkey, op), c = best
            if c != self.colour:
                pt = self.swatches.get(c) or self._find_colour(c)
                if pt is not None and 6 in avail:
                    return (6, pt)
            else:
                route = self._route(key, bkey)
                if route is not None:
                    return route[0] if route else op
        # something clickable next to the cursor that might paint (a stamp), at a
        # cursor place seen this level, or at one not seen yet
        if 6 in avail:
            step = self._untried_blob()
            if step is not None:
                return step
            return self._unseen_place()
        return None

    def _unseen_place(self) -> tuple | None:
        """The next step towards a cursor place not seen this level."""
        key = self._key(self.grid)
        places = set(self.tried) | {k for e in self.edges.values() for k in e.values()}
        for k in sorted(places - set(self.blobs_seen)):
            route = self._route(key, k)
            if route:
                return route[0]
        return None

    def _rare_colour(self, share: float = 0.5) -> tuple[int, int] | None:
        """A click selecting the palette colour the canvas has least of, when the
        selected colour covers more than ``share`` of the canvas and more than
        that colour does; else None."""
        canvas = self._cut(self.grid, self.canvas)
        mine = int((canvas == self.colour).sum()) if self.colour is not None else canvas.size
        if mine <= share * canvas.size:
            return None
        best = None
        for c in sorted(self._palette()):
            n = int((canvas == c).sum())
            if c != self.colour and n < mine and (best is None or n < best[0]):
                best = (n, c)
        if best is None:
            return None
        return self.swatches.get(best[1]) or self._find_colour(best[1])

    def _outside(self, x: int, y: int) -> bool:
        for w in (self.canvas, self.reference):
            wx, wy, ww, hh = w
            if wx <= x < wx + ww and wy <= y < wy + hh:
                return False
        return not (self.mask is not None and self.mask[y, x])

    def _palette(self) -> set[int]:
        """Colours that can be selected: clicked before, or shown as a small patch
        outside the canvas and the reference."""
        out = set(self.swatches)
        target = self._cut(self.grid, self.reference)
        for c in np.unique(target).tolist():
            if c not in out and self._find_colour(int(c)) is not None:
                out.add(int(c))
        return out

    def _find_colour(self, c: int) -> tuple[int, int] | None:
        """A small patch of colour ``c`` outside the canvas and the reference (a
        palette square) that has not been clicked in vain, as a click point."""
        best = None
        for comp in fp.segment(fp.apply_mask(self.grid, self.mask)):
            if comp.color != c or comp.area > STAMP_AREA:
                continue
            x, y = comp.click_point()
            if not self._outside(x, y) or (x, y) in self.dead:
                continue
            if best is None or comp.area > best[0]:
                best = (comp.area, (x, y))
        return best[1] if best else None

    def _blobs(self, grid: np.ndarray) -> list[tuple[int, int]]:
        """Click points of the small things round the canvas other than the
        cursor (stamps drawn next to it), one per thing."""
        x, y, w, h = self.canvas
        x0, y0 = max(x - MARGIN, 0), max(y - MARGIN, 0)
        x1, y1 = min(x + w + MARGIN, grid.shape[1]), min(y + h + MARGIN, grid.shape[0])
        m = grid[y0:y1, x0:x1] != self.bg
        m[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
        if self.mask is not None:
            m &= ~self.mask[y0:y1, x0:x1]
        comps = sorted((c for c in fp.segment(m.astype(np.int8)) if c.color == 1), key=lambda c: -c.area)
        out = []
        for c in comps[1:]:  # the largest is the cursor
            if 4 <= c.area <= STAMP_AREA:
                cx, cy = c.click_point()
                out.append((cx + x0, cy + y0))
        return out

    def _untried_blob(self) -> tuple | None:
        """The next step towards clicking a thing seen next to the cursor this
        level and not clicked at that cursor place yet."""
        key = self._key(self.grid)
        for k in [key] + sorted(self.blobs_seen):
            pts = [pt for pt in self.blobs_seen.get(k, []) if (6, pt) not in self.tried.get(k, set())]
            if not pts:
                continue
            if k == key:
                # in the colour the canvas has least of, so that a stamp shows
                return (6, self._rare_colour(0.0) or pts[0])
            route = self._route(key, k)
            if route:
                return route[0]
        return None

    # -- acting -----------------------------------------------------------------

    def _give_up(self, why: str) -> None:
        self.stats["give_up_" + why] += 1
        self.gave_up = True
        self.in_control = False
        self.strikes += 1
        if self.strikes >= (3 if self.confirmed else 2):
            self.disabled = True

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if self.canvas is None:
            return None  # not recognised: nothing has painted a picture-sized canvas yet
        if not self.in_control:
            self.in_control = True
            self.stats["takeovers"] += 1
        limit = MAX_LEVEL_ACTIONS if self.confirmed else UNPROVEN_ACTIONS
        if self.level_actions >= limit:
            self._give_up("actions")
            return None
        if self.deaths >= 2:
            self._give_up("deaths")
            return None
        step = None
        if not self.confirmed and not self.surveyed:
            # first level: try every brush once, later levels need them
            step = self._survey(obs)
            if step is None:
                self.surveyed = True
            else:
                self.stats["survey_steps"] += 1
        if step is None and self.new_stamp and 6 in obs.available_actions:
            # a stamp appeared: try every stamp next to every cursor place
            step = self._untried_blob() or self._unseen_place()
            if step is not None:
                self.stats["stamp_survey_steps"] += 1
        if step is None:
            plan = self._make_plan()
            if plan is None:
                # a stamp seen this level and not tried may complete an exact plan
                step = self._untried_blob() if 6 in obs.available_actions else None
                if step is not None:
                    plan = [step]
                    self.stats["stamp_steps"] += 1
                else:
                    plan = self._make_plan(approximate=True)
            if plan == []:
                # everything checked already matches and the level goes on: what
                # is left is unknown to the solver
                self._give_up("matched")
                return None
            if plan is None:
                step = self._explore(obs)
                if step is None:
                    self._give_up("nothing_to_try")
                    return None
                self.stats["explore_steps"] += 1
            else:
                step = plan[0]
                self.stats["plan_steps"] += 1
        self.level_actions += 1
        a, pt = step
        if a == 6:
            return GameAction.ACTION6, {"x": pt[0], "y": pt[1]}
        return GameAction.from_id(a), None
