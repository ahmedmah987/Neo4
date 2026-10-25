"""The spell family: walk the avatar to the exit, casting the spells drawn on a click pad.

Some puzzles show a maze, an avatar (a square in two colours, split across or down its
middle), an exit (a box drawn in the avatar's two colours) and, apart, a pad of 3x3 small
cells that clicks toggle. When the cells toggled on make one of the level's spells, it is
cast and the pad clears: the avatar shrinks or grows (and passes a narrow gap, or not), or
jumps to a marker, or something leaves it along the way it last moved, and if that meets a
target, the target and the barriers of the colour at its core are gone. On the first levels
the pad shows the spell to draw, its cells in a colour of their own; later levels show
none, and the spells learned on the first are cast from memory. A move takes the avatar its
own size along (a large one that is blocked tries half that); it ends the level when it
reaches the exit (a large one whose full step touches the exit is taken to try the half
step, until a level is seen to end on the full step).

From the frames alone:

* the avatar is the one square made of two equal rectangles of two colours; its colours
  are kept for the game. The floor is the colour round it (a colour on the screen's edges is
  taken last), the wall the commonest other colour, the exit the box of the avatar's colours
  elsewhere;
* the pad is nine equal one-colour squares on a 3x3 lattice; the spell shown is the cells
  of the colour fewer of them have, and a cell of a third colour is toggled on;
* what a spell does is learned when it is first cast, from the avatar and the screen before
  and after: a new size (and the size it gives back, until a cast at that size is seen to
  change nothing), a new place (a jump), cells that
  became floor (a shot), or nothing (a shot that met nothing on its way: tried again aimed
  elsewhere). From the first shot that clears something, a target is a box framed in the
  colour of the box it cleared, and what it clears is its core's colour, wherever it is;
  the colours its rows crossed on the way to the target are ones a shot flies through, and
  the colours the avatar was seen standing on after a move or a grow are ones it can walk;
* where a jump lands is the mark found there before it (the square of the avatar's size, as
  drawn), kept for the size, marks among the controls aside; where several such marks are
  drawn, the next is the one with the ring that was round the first (what is neither floor
  nor wall just outside it), and the avatar on the mark it jumped to hides that mark.

The plan: a search over the avatar's place, size and facing, the barrier colours cleared
and the jumps made at each size, for the fewest actions to the exit, by moves and by casts
whose outcome is known (a cast costs a click a cell). A shot is taken to clear a target
only when every row (or column) of the avatar meets that target first. When there is no
plan, a spell never cast is cast to see what it does, a jump whose landing is not known is
taken from the cheapest place at its size, and a shot with no known target is aimed by a
move (or a move into a wall) the way that meets something that is neither wall nor floor.
"""

from __future__ import annotations

import heapq
from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
MAX_FAILS = 10
MAX_LEVEL_ACTIONS = 300
MAX_STATES = 300_000


def find_avatar(comps: list, look: tuple | None = None):
    """(x, y, size, colours) of the one square made of two equal rectangles of two colours."""
    rects = [c for c in comps if c.is_rectangle and c.area <= 50]
    at = {(c.x1, c.y1): c for c in rects}
    found = {}
    for a in rects:
        for b in (at.get((a.x1, a.y2 + 1)), at.get((a.x2 + 1, a.y1))):
            if b is None or b.color == a.color or (b.width, b.height) != (a.width, a.height):
                continue
            if b.y1 == a.y2 + 1 and b.x1 == a.x1 and a.width == 2 * a.height:
                s = a.width
            elif b.x1 == a.x2 + 1 and b.y1 == a.y1 and a.height == 2 * a.width:
                s = a.height
            else:
                continue
            if look is not None and {a.color, b.color} != set(look):
                continue
            found[(a.x1, a.y1, s)] = (a.color, b.color)
    if len(found) != 1:
        return None
    (x, y, s), cols = next(iter(found.items()))
    return x, y, s, cols


def window_sums(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """The number of set cells in every h x w window, by its top left (rows y, columns x)."""
    H, W = mask.shape
    m = np.zeros((H + 1, W + 1), dtype=np.int32)
    m[1:, 1:] = mask.astype(np.int32).cumsum(0).cumsum(1)
    return m[h:, w:] - m[:H + 1 - h, w:] - m[h:, :W + 1 - w] + m[:H + 1 - h, :W + 1 - w]


def fits(mask: np.ndarray, s: int) -> np.ndarray:
    """free[y, x]: the s x s square at (x, y) lies wholly on the mask."""
    return window_sums(mask, s, s) == s * s


def halves(grid: np.ndarray, look: tuple, s: int) -> list:
    """Every s x s square made of two halves, one of each of the two colours: [(x, y)]."""
    H, W = grid.shape
    if H < s or W < s:
        return []
    h = s // 2
    n = H - s + 1, W - s + 1
    out = set()
    for a, b in (look, look[::-1]):
        A, B = grid == a, grid == b
        across = (window_sums(A, h, s)[:n[0], :n[1]] == h * s) & (window_sums(B, h, s)[h:h + n[0], :n[1]] == h * s)
        down = (window_sums(A, s, h)[:n[0], :n[1]] == h * s) & (window_sums(B, s, h)[:n[0], h:h + n[1]] == h * s)
        ys, xs = np.nonzero(across | down)
        out.update(zip(xs.tolist(), ys.tolist()))
    return sorted(out)


def find_pad(comps: list):
    """The nine cells of a 3x3 lattice of equal one-colour squares: [(x, y, size)] by rows."""
    squares = [c for c in comps if c.is_rectangle and c.width == c.height and 2 <= c.width <= 6]
    at = {(c.x1, c.y1, c.width) for c in squares}
    pads = []
    for c in squares:
        k = c.width
        for p in range(k + 1, 3 * k + 4):
            cells = [(c.x1 + i * p, c.y1 + j * p, k) for j in range(3) for i in range(3)]
            if all(cell in at for cell in cells):
                pads.append(cells)
    return pads[0] if len(pads) == 1 else None


def find_targets(grid: np.ndarray, comps: list, frame: int) -> list:
    """Boxes framed in the colour with a core of one other colour: [((x1, y1, x2, y2), core)]."""
    out = []
    for c in comps:
        if c.color != frame or c.width < 3 or c.height < 3:
            continue
        if c.area != 2 * c.width + 2 * c.height - 4:
            continue  # not a ring round its box
        core = np.unique(grid[c.y1 + 1:c.y2, c.x1 + 1:c.x2])
        if len(core) == 1 and int(core[0]) != frame:
            out.append(((c.x1, c.y1, c.x2, c.y2), int(core[0])))
    return out


def find_marks(grid: np.ndarray, sig: tuple) -> list:
    """Every (x, y) at which the signature's cells ((dx, dy, colour), ...) are all drawn."""
    H, W = grid.shape
    if not sig:
        return []
    lo_x = min(d[0] for d in sig)
    lo_y = min(d[1] for d in sig)
    hi_x = max(d[0] for d in sig)
    hi_y = max(d[1] for d in sig)
    xs = range(max(0, -lo_x), W - max(0, hi_x))
    ys = range(max(0, -lo_y), H - max(0, hi_y))
    if not xs or not ys:
        return []
    ok = np.ones((len(ys), len(xs)), dtype=bool)
    for dx, dy, c in sig:
        ok &= grid[ys.start + dy:ys.stop + dy, xs.start + dx:xs.stop + dx] == c
    py, px = np.nonzero(ok)
    return [(int(x) + xs.start, int(y) + ys.start) for y, x in zip(py, px)]


def blob(mask: np.ndarray, x: int, y: int) -> set:
    """The 8-connected cells of the mask that hold (x, y)."""
    H, W = mask.shape
    out = {(x, y)}
    q = deque([(x, y)])
    while q:
        cx, cy = q.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (cx + dx, cy + dy)
                if 0 <= n[0] < W and 0 <= n[1] < H and n not in out and mask[n[1], n[0]]:
                    out.add(n)
                    q.append(n)
    return out


class Scene:
    """What a level shows: the avatar, the floor and wall colours, the exit and the pad."""

    def __init__(self, grid: np.ndarray, avatar, pad) -> None:
        x, y, s, look = avatar
        self.look = look
        ring = Counter()
        H, W = grid.shape
        for i in range(-1, s + 1):
            for p in ((x + i, y - 1), (x + i, y + s), (x - 1, y + i), (x + s, y + i)):
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    ring[int(grid[p[1], p[0]])] += 1
        for c in look:
            ring.pop(c, None)
        # the floor is round the avatar; the colours on the screen's edges are taken last
        edge = set(grid[0].tolist()) | set(grid[-1].tolist()) | set(grid[:, 0].tolist()) | set(grid[:, -1].tolist())
        self.floor = max(ring, key=lambda c: (c not in edge, ring[c])) if ring else None
        rest = Counter(int(v) for v in grid.ravel())
        for c in (self.floor,) + tuple(look):
            rest.pop(c, None)
        self.wall = rest.most_common(1)[0][0] if rest else None
        self.pad = pad
        self.pad_frame = int(grid[pad[0][1], pad[0][0] - 1]) if pad[0][0] > 0 else None
        xs = [c[0] for c in pad]
        ys = [c[1] for c in pad]
        k = pad[0][2]
        self.pad_box = (min(xs) - 1, min(ys) - 1, max(xs) + k, max(ys) + k)
        colours = [int(grid[cy, cx]) for cx, cy, _ in pad]
        n = Counter(colours)
        self.blank = n.most_common(1)[0][0]
        self.hint_colour = n.most_common()[1][0] if len(n) == 2 else None
        self.hint = frozenset(i for i, c in enumerate(colours) if c == self.hint_colour) if self.hint_colour is not None else frozenset()
        # the exit: the box of the avatar's colours that is not the avatar
        mask = np.isin(grid, look)
        mask[y:y + s, x:x + s] = False
        self.exit = None
        ys_, xs_ = np.nonzero(mask)
        if len(xs_):
            cells = blob(mask, int(xs_[0]), int(ys_[0]))
            if len(cells) == int(mask.sum()):
                bx = [c[0] for c in cells]
                by = [c[1] for c in cells]
                box = (min(bx), min(by), max(bx), max(by))
                if (box[2] - box[0] + 1) * (box[3] - box[1] + 1) == len(cells):
                    self.exit = box

    def selected(self, grid: np.ndarray) -> frozenset:
        return frozenset(i for i, (cx, cy, _) in enumerate(self.pad)
                         if int(grid[cy, cx]) not in (self.blank, self.hint_colour))

    def click(self, i: int) -> dict:
        cx, cy, k = self.pad[i]
        return {"x": cx + k // 2, "y": cy + k // 2}


class Model:
    """The screen as the search sees it: what is walkable once some barrier colours are
    cleared, where a square of each size fits, what a shot meets."""

    def __init__(self, solver: "SpellSolver") -> None:
        self.solver = solver
        sc = solver.scene
        g = solver.grid
        self.grid = g
        base = np.isin(g, [sc.floor, *solver.walkable])
        x, y, s = solver.avatar
        base[y:y + s, x:x + s] = True
        x1, y1, x2, y2 = sc.pad_box
        base[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = False
        self.base = base
        self.targets = solver.targets
        self.through = np.isin(g, list(solver.shot_through)) if solver.shot_through else None
        self.masks: dict = {}
        self.free: dict = {}
        self.shots: dict = {}

    def walkable(self, cleared: frozenset) -> np.ndarray:
        if cleared not in self.masks:
            m = self.base.copy()
            if cleared:
                m |= np.isin(self.grid, list(cleared))
                for (x1, y1, x2, y2), core in self.targets:
                    if core in cleared:
                        m[y1:y2 + 1, x1:x2 + 1] = True
            self.masks[cleared] = m
        return self.masks[cleared]

    def fits(self, x: int, y: int, s: int, cleared: frozenset) -> bool:
        key = (s, cleared)
        if key not in self.free:
            self.free[key] = fits(self.walkable(cleared), s)
        f = self.free[key]
        return 0 <= y < f.shape[0] and 0 <= x < f.shape[1] and bool(f[y, x])

    def shot(self, x: int, y: int, s: int, d, cleared: frozenset):
        """The target a shot from here clears: the one every row (or column) meets first."""
        key = (x, y, s, d, cleared)
        if key in self.shots:
            return self.shots[key]
        m = self.walkable(cleared)
        if self.through is not None:
            m = m | self.through
        H, W = m.shape
        lanes = [(x + s - 1 if d[0] > 0 else x, y + i) for i in range(s)] if d[0] else \
                [(x + i, y + s - 1 if d[1] > 0 else y) for i in range(s)]
        hit = None
        for px, py in lanes:
            while True:
                px, py = px + d[0], py + d[1]
                if not (0 <= px < W and 0 <= py < H) or not m[py, px]:
                    break
            t = next((t for t in self.targets if t[1] not in cleared
                      and t[0][0] <= px <= t[0][2] and t[0][1] <= py <= t[0][3]), None)
            if t is None or (hit is not None and t != hit):
                hit = None
                break
            hit = t
        self.shots[key] = hit
        return hit


# --- the solver --------------------------------------------------------------------


class SpellSolver:
    name = "spells"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.look: tuple | None = None
        self.spells: dict = {}  # pattern -> "resize" | "jump" | "shot" | "dud", learned for the game
        self.sizes: dict = {}  # size -> the size a resize gives
        self.sizes_guessed: set = set()  # sizes whose resize is only taken to undo one seen
        self.marks: dict = {}  # size -> (the mark a jump lands on, the ring round it)
        self.target_frame: int | None = None  # the colour a target is framed in
        self.shot_through: set = set()  # colours a shot was seen to fly through
        self.walkable: set = set()  # colours, besides the floor, the avatar was seen standing on
        self.half: bool | None = None  # a blocked large move tries half the step
        self.exit_full = False  # a full step that touches the exit ends the level (seen once)
        self.facing = None  # the way the avatar last moved (or tried to)
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.scene: Scene | None = None
        self.queue: list = []
        self.expect = None
        self.actions = 0
        self.fails = 0
        self.aimed: set = set()  # (pattern, what the shot's way meets) tried without effect
        self.stops: set = set()  # (x, y, size) a move was seen not to reach
        self.banned: set = set()  # patterns that cast nothing on this level
        self.landed: dict = {}  # size -> where the jumps of this level landed, in turn
        self.failed: set = set()  # casts (pattern, x, y, size, facing) seen to change nothing here

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    # --- reading ---------------------------------------------------------------------

    def _setup(self, grid: np.ndarray) -> bool:
        comps = fp.segment(grid)
        av = find_avatar(comps, self.look)
        pad = find_pad(comps)
        if av is None or pad is None:
            return False
        sc = Scene(grid, av, pad)
        if sc.exit is None or sc.floor is None:
            return False
        if not sc.hint and not any(k in ("resize", "jump", "shot") for k in self.spells.values()):
            return False  # nothing to cast: not seen yet, and not shown
        self.scene = sc
        self.found_look = av[3]
        self.avatar = av[:3]
        self.grid = grid
        return True

    def _read(self, grid: np.ndarray) -> None:
        self.grid = grid
        comps = fp.segment(grid)
        self.targets = find_targets(grid, comps, self.target_frame) if self.target_frame is not None else []
        # the controls (the pad, the spells' icons) are boxes framed in the pad's frame colour
        f = self.scene.pad_frame
        self.controls = [(c.x1, c.y1, c.x2, c.y2) for c in comps if c.color == f and c.width >= 6 and c.height >= 6]

    def _avatar_now(self, grid: np.ndarray, guess=None):
        """The avatar by its look: the squares of two halves in its colours, those inside the
        exit and those inside a bigger one left out; the one expected first."""
        sc = self.scene
        x1, y1, x2, y2 = sc.exit
        found = []
        for s in (8, 6, 4, 2):
            for x, y in halves(grid, sc.look, s):
                if x1 <= x and x + s - 1 <= x2 and y1 <= y and y + s - 1 <= y2:
                    continue
                if any(fx <= x and x + s <= fx + fs and fy <= y and y + s <= fy + fs for fx, fy, fs in found):
                    continue
                found.append((x, y, s))
        if guess is not None and guess in found:
            return guess
        return found[0] if len(found) == 1 else None

    def _castable(self) -> list:
        sc = self.scene
        out = [sc.hint] if sc.hint else []
        out += [p for p, k in self.spells.items() if k in ("resize", "jump", "shot") and p not in out]
        return [p for p in out if p not in self.banned]

    def _landings(self, s: int):
        """Where the jumps from now at size s land, in turn: a list, or None if not known."""
        if s not in self.marks:
            return None
        core, ring = self.marks[s]

        def placed(found):
            return [(x, y) for x, y in found
                    if not any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in self.controls)]

        marks = placed(find_marks(self.grid, core))
        x, y, size = self.avatar
        last = self.landed.get(s)
        if size == s and last and last[-1] == (x, y) and (x, y) not in marks:
            marks.append((x, y))  # standing on the mark it jumped to, and hiding it
        if not marks:
            return []
        if len(marks) == 1:
            return marks * 4
        nxt = [m for m in marks if m in set(placed(find_marks(self.grid, core + ring)))] if ring else []
        if len(nxt) != 1:
            return None
        if len(marks) == 2:
            other = next(m for m in marks if m != nxt[0])
            return [nxt[0], other] * 2
        return nxt

    # --- the model -------------------------------------------------------------------

    def _at_exit(self, x: int, y: int, s: int) -> bool:
        x1, y1, x2, y2 = self.scene.exit
        return x < x2 + 1 and x + s > x1 and y < y2 + 1 and y + s > y1

    def _step(self, model: Model, x, y, s, d, cleared):
        """The avatar's place after a key: (x, y), or "exit" when it reaches the exit."""
        tries = [s]
        if s >= 4 and self.half is not False:
            tries.append(s // 2)
        for i, n in enumerate(tries):
            nx, ny = x + d[0] * n, y + d[1] * n
            if self._at_exit(nx, ny, s):
                if i + 1 < len(tries) and not self.exit_full:
                    continue  # touching the exit blocks the full step: the half step decides
                return "exit"
            if (nx, ny, s) not in self.stops and model.fits(nx, ny, s, cleared):
                return (nx, ny)
        return (x, y)

    def _resize(self, model: Model, x, y, s, cleared):
        n = self.sizes.get(s)
        if n is None:
            return None
        if n < s:
            return (x, y, n)
        g = n - s
        for dx in range(-g, 1):
            for dy in range(-g, 1):
                if model.fits(x + dx, y + dy, n, cleared):
                    return (x + dx, y + dy, n)
        return None

    def _ray(self, st, d, passable):
        """What lies first along the way d from the avatar: ((x, y, colour) per row or column)."""
        x, y, s = st
        g = self.grid
        H, W = g.shape
        out = []
        lanes = [(x + s - 1 if d[0] > 0 else x, y + i) for i in range(s)] if d[0] else \
                [(x + i, y + s - 1 if d[1] > 0 else y) for i in range(s)]
        for px, py in lanes:
            hit = None
            while True:
                px, py = px + d[0], py + d[1]
                if not (0 <= px < W and 0 <= py < H):
                    break
                if not passable[py, px]:
                    hit = (px, py, int(g[py, px]))
                    break
            out.append(hit)
        return tuple(out)

    def _meets(self, ray) -> bool:
        return any(h is not None and h[2] != self.scene.wall for h in ray)

    def _clicks(self, pattern, selected, final):
        """The clicks that turn the pad's cells on into the pattern, with what each shows."""
        out = []
        sel = set(selected)
        cells = sorted(set(pattern) ^ sel)
        for n, i in enumerate(cells):
            sel ^= {i}
            last = n == len(cells) - 1
            out.append(((GameAction.ACTION6, self.scene.click(i)), ("cast", pattern, final) if last else ("pad", frozenset(sel))))
        return out

    def _search(self):
        """The fewest actions to the exit: [(action, prediction)], or an exploration, or None."""
        sc = self.scene
        model = Model(self)
        selected = sc.selected(self.grid)
        casts = self._castable()
        sizes = sorted({self.avatar[2], *self.sizes.keys(), *self.sizes.values()})
        landings = {s: self._landings(s) for s in sizes}
        x0, y0, s0 = self.avatar
        start = (x0, y0, s0, self.facing, frozenset(), (0,) * len(sizes))
        cost = {start: 0}
        prev = {start: None}
        heap = [(0, 0, start)]
        tick = 0
        win = None
        while heap:
            g, _, st = heapq.heappop(heap)
            if g > cost.get(st, 1 << 30):
                continue
            if win is not None and g >= win[0]:
                break
            x, y, s, f, cleared, jumps = st
            nexts = []
            for k, d in KEYS.items():
                r = self._step(model, x, y, s, d, cleared)
                if r == "exit":
                    if win is None or g + 1 < win[0]:
                        win = (g + 1, st, ("key", k))
                    continue
                nexts.append((g + 1, (r[0], r[1], s, d, cleared, jumps), ("key", k)))
            for p in casts:
                if (p, x, y, s, f if self.spells.get(p) == "shot" else None) in self.failed:
                    continue
                c = g + len(set(p) ^ (selected if st == start else frozenset()))
                kind = self.spells.get(p)
                if kind == "resize":
                    r = self._resize(model, x, y, s, cleared)
                    if r is not None:
                        nexts.append((c, (r[0], r[1], r[2], f, cleared, jumps), ("cast", p)))
                elif kind == "jump":
                    i = sizes.index(s)
                    land = landings.get(s)
                    if land and jumps[i] < len(land):
                        lx, ly = land[jumps[i]]
                        js = jumps[:i] + (jumps[i] + 1,) + jumps[i + 1:]
                        nexts.append((c, (lx, ly, s, f, cleared, js), ("cast", p)))
                elif kind == "shot" and f is not None:
                    t = model.shot(x, y, s, f, cleared)
                    if t is not None:
                        nexts.append((c, (x, y, s, f, cleared | {t[1]}, jumps), ("cast", p)))
            for c, r, how in nexts:
                if c < cost.get(r, 1 << 30):
                    cost[r] = c
                    prev[r] = (st, how)
                    tick += 1
                    heapq.heappush(heap, (c, tick, r))
            if len(cost) > MAX_STATES:
                self.stats["search_too_big"] += 1
                break
        if win is not None:
            _, st, last = win
            return self._unroll(prev, st, [(last, "exit")], selected)
        return self._explore(model, prev, cost, casts, sizes, landings, selected)

    def _unroll(self, prev, st, tail, selected):
        steps = list(tail)
        while prev[st] is not None:
            p, how = prev[st]
            steps.append((how, st))
            st = p
        steps.reverse()
        out = []
        sel = selected
        for how, after in steps:
            if how[0] == "key":
                out.append(((how[1], None), ("at", after if after == "exit" else after[:3])))
            else:
                out.extend(self._clicks(how[1], sel, after[:3]))
                sel = frozenset()
        return out

    def _explore(self, model, prev, cost, casts, sizes, landings, selected):
        """Cast a spell to see what it does: one never cast at once; a jump whose landing is
        not known from the cheapest place at its size; a shot aimed at something."""
        for p in casts:
            if p not in self.spells:
                self.stats["explore_cast"] += 1
                return self._clicks(p, selected, None)
        best = None
        for p in casts:
            kind = self.spells.get(p)
            if kind == "jump":
                for st, g in cost.items():
                    if landings.get(st[2]) is None and (best is None or g + len(p) < best[0]):
                        best = (g + len(p), st, None, p, None)
            if kind == "dud" or (kind == "shot" and self.target_frame is None):
                passable = model.walkable(frozenset())
                for st, g in cost.items():
                    x, y, s, f, cleared, _ = st
                    if cleared:
                        continue
                    if f is not None:
                        ray = self._ray((x, y, s), f, passable)
                        if self._meets(ray) and (p, ray) not in self.aimed and (best is None or g < best[0]):
                            best = (g, st, None, p, ray)
                    for k, d in KEYS.items():
                        r = self._step(model, x, y, s, d, cleared)
                        if r == "exit":
                            continue
                        ray = self._ray((r[0], r[1], s), d, passable)
                        if self._meets(ray) and (p, ray) not in self.aimed and (best is None or g + 1 < best[0]):
                            best = (g + 1, st, (k, (r[0], r[1], s)), p, ray)
        if best is None:
            return None
        _, st, key, p, ray = best
        self.stats["explore_jump" if ray is None else "explore_aim"] += 1
        plan = self._unroll(prev, st, [], selected) if prev.get(st) is not None else []
        sel = frozenset() if any(pred[0] == "cast" for _, pred in plan) else selected
        if key is not None:
            plan.append(((key[0], None), ("at", key[1])))
        plan.extend(self._clicks(p, sel, None))
        self.aim = (p, ray) if ray is not None else None
        return plan

    # --- watching --------------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        if action in KEYS:
            self.facing = KEYS[action]
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
                if self.scene is not None:
                    self.look = self.found_look
                    if self.expect is not None and self.expect[2][0] == "at" and self.expect[2][1] != "exit" and action in KEYS:
                        # the level ended on a move the model sent short of the exit
                        x, y, sz = self.expect[0]
                        d = KEYS[action]
                        if self._at_exit(x + d[0] * sz, y + d[1] * sz, sz):
                            self.exit_full = True
                            self.stats["learned_exit_full"] += 1
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        exp, self.expect = self.expect, None
        if action is None or self.scene is None:
            return
        if action is GameAction.RESET or exp is None:
            self.scene = None
            self.read = False
            self.queue = []
            return
        grid = np.asarray(obs.frame[-1])
        before, before_grid, pred = exp
        guess = pred[1] if pred[0] == "at" else (pred[2] if pred[0] == "cast" and pred[2] else before)
        now = self._avatar_now(grid, guess)
        if now is None:
            self._fail("avatar_lost")
            return
        sc = self.scene
        sel = sc.selected(grid)
        if pred == ("at", "exit"):
            self._fail("exit_missed")
        elif pred[0] == "at":
            self._learn_ground(now, before_grid)
            if now != pred[1]:
                x, y, s = before
                if now == before and pred[1] != before:
                    self.stops.add(pred[1])
                    self.stats["stopped"] += 1
                elif s >= 4 and now[2] == s and (abs(now[0] - x) + abs(now[1] - y)) * 2 == s:
                    self.half = True
                elif s >= 4 and now == before:
                    self.half = False
                self._fail("mismatch")
        elif pred[0] == "pad":
            if sel != pred[1]:
                self._fail("pad_mismatch")
        elif pred[0] == "cast":
            _, pattern, final = pred
            if sel:
                self.banned.add(pattern)
                self._fail("no_cast")
            else:
                gone = self._gone(before, before_grid, grid)
                self._learn_cast(pattern, before, now, before_grid, gone)
                cleared_as_planned = final is None or self.spells.get(pattern) != "shot" or gone.any()
                if final is None or now != final or not cleared_as_planned:
                    if final is not None:
                        self.stats["cast_mismatch"] += 1
                    self.queue = []
        self.avatar = now
        self._read(grid)

    def _gone(self, before, g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
        """Cells that became floor (the pad and the avatar's place aside)."""
        sc = self.scene
        gone = (g1 == sc.floor) & (g0 != sc.floor)
        x1, y1, x2, y2 = sc.pad_box
        gone[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = False
        x, y, s = before
        gone[y:y + s, x:x + s] = False
        return gone

    def _learn_cast(self, pattern, before, now, g0: np.ndarray, gone: np.ndarray) -> None:
        sc = self.scene
        x, y, s = before
        if now[2] != s:
            self.spells[pattern] = "resize"
            self.sizes[s] = now[2]
            self.sizes_guessed.discard(s)
            if now[2] not in self.sizes:
                self.sizes[now[2]] = s  # taken to undo itself until seen not to
                self.sizes_guessed.add(now[2])
            self._learn_ground(now, g0)
            self.stats["learned_resize"] += 1
            return
        if now[:2] != (x, y):
            self.spells[pattern] = "jump"
            self._learn_mark(now, g0)
            self.landed.setdefault(s, []).append(now[:2])
            self.stats["learned_jump"] += 1
            return
        if gone.any():
            self.spells[pattern] = "shot"
            box = self._learn_target(g0, gone)
            if box is not None and self.facing is not None:
                self._learn_through(before, self.facing, box, g0)
            self.stats["learned_shot"] += 1
            return
        kind = self.spells.get(pattern)
        if kind in ("resize", "jump", "shot"):
            # a grow with no room, a shot that missed: what it does is known, not to be
            # tried here again; a resize only guessed to undo itself is not one
            self.stats["cast_blocked"] += 1
            if kind == "resize" and s in self.sizes_guessed:
                del self.sizes[s]
                self.sizes_guessed.discard(s)
                self.stats["unlearned_resize"] += 1
            self.failed.add((pattern, x, y, s, self.facing if kind == "shot" else None))
            return
        self.spells[pattern] = "dud"
        aim = getattr(self, "aim", None)
        if aim is not None and aim[0] == pattern:
            self.aimed.add(aim)
        elif self.facing is not None:
            self.aimed.add((pattern, self._ray(before, self.facing, Model(self).walkable(frozenset()))))
        self.stats["cast_no_effect"] += 1

    def _learn_mark(self, now, g0: np.ndarray) -> None:
        """The mark a jump of this size landed on, and the ring round it, from the screen before."""
        sc = self.scene
        x, y, s = now
        H, W = g0.shape
        core, ring = [], []
        for dy in range(-1, s + 1):
            for dx in range(-1, s + 1):
                px, py = x + dx, y + dy
                if not (0 <= px < W and 0 <= py < H):
                    continue
                c = int(g0[py, px])
                if 0 <= dx < s and 0 <= dy < s:
                    core.append((dx, dy, c))  # all of it, floor too: what lies inside tells marks apart
                elif c not in (sc.floor, sc.wall):
                    ring.append((dx, dy, c))
        old = self.marks.get(s)
        if old is not None:
            # the same mark seen again keeps what the two sightings share
            core = [p for p in core if p in old[0]] or old[0]
            ring = [p for p in ring if p in old[1]]
        self.marks[s] = (tuple(core), tuple(ring))

    def _learn_ground(self, now, g0: np.ndarray) -> None:
        """The colours the avatar now stands on (as drawn before it got there) do not stop it."""
        sc = self.scene
        x, y, s = now
        for c in np.unique(g0[y:y + s, x:x + s]).tolist():
            if c not in (sc.floor, *sc.look) and c not in self.walkable:
                self.walkable.add(int(c))
                self.stats["learned_walkable"] += 1

    def _learn_through(self, before, d, box, g0: np.ndarray) -> None:
        """The colours a clearing shot's rows (or columns) crossed on their way to the target
        are ones a shot flies through."""
        sc = self.scene
        x, y, s = before
        x1, y1, x2, y2 = box
        H, W = g0.shape
        lanes = [(x + s - 1 if d[0] > 0 else x, y + i) for i in range(s)] if d[0] else \
                [(x + i, y + s - 1 if d[1] > 0 else y) for i in range(s)]
        for px, py in lanes:
            crossed = set()
            while True:
                px, py = px + d[0], py + d[1]
                if not (0 <= px < W and 0 <= py < H) or g0[py, px] == sc.wall:
                    crossed = set()  # this row met a wall first: it tells nothing
                    break
                if x1 <= px <= x2 and y1 <= py <= y2:
                    break
                c = int(g0[py, px])
                if c != sc.floor:
                    crossed.add(c)
            for c in crossed - self.shot_through:
                self.shot_through.add(c)
                self.stats["learned_shot_through"] += 1

    def _learn_target(self, g0: np.ndarray, gone: np.ndarray):
        """A shot that cleared cells: the box among them framed in one colour round a core of
        another, the rest of what went being of the core's colour, makes targets known.
        Returns the box."""
        ys, xs = np.nonzero(gone)
        seen: set = set()
        for x, y in zip(xs.tolist(), ys.tolist()):
            if (x, y) in seen:
                continue
            cells = blob(gone, x, y)
            seen |= cells
            bx = [c[0] for c in cells]
            by = [c[1] for c in cells]
            x1, y1, x2, y2 = min(bx), min(by), max(bx), max(by)
            if x2 - x1 < 2 or y2 - y1 < 2 or len(cells) != (x2 - x1 + 1) * (y2 - y1 + 1):
                continue
            box = g0[y1:y2 + 1, x1:x2 + 1]
            border = set(box[0].tolist()) | set(box[-1].tolist()) | set(box[:, 0].tolist()) | set(box[:, -1].tolist())
            core = set(box[1:-1, 1:-1].ravel().tolist())
            if len(border) == 1 and len(core) == 1 and border != core:
                rest = gone.copy()
                rest[y1:y2 + 1, x1:x2 + 1] = False
                if set(g0[rest].tolist()) <= core and (self.target_frame is None or self.target_frame in border):
                    if self.target_frame is None:
                        self.target_frame = border.pop()
                        self.stats["learned_target"] += 1
                    return (x1, y1, x2, y2)
        return None

    def _fail(self, why: str) -> None:
        self.fails += 1
        self.stats[why] += 1
        self.queue = []

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4, 6} <= set(obs.available_actions):
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if self.scene is None:
            if self.read:
                return None
            self.read = True
            if not self._setup(grid):
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
        self._read(grid)
        if not self.queue:
            self.aim = None
            plan = self._search()
            if not plan:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            self.queue = plan
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        (a, d), pred = self.queue.pop(0)
        self.expect = (self.avatar, grid, pred)
        self.actions += 1
        self.stats["actions"] += 1
        return a, d
