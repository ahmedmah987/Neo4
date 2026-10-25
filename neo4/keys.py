"""The key family: carry a key that matches the door into it, changing the key on tiles.

Some puzzles show a maze of cells, an avatar, a door with a glyph in it, and, in a box apart,
the glyph of the key the avatar carries (drawn at twice the size). Tiles on the floor
change the key when the avatar steps on them: one turns it a quarter, one gives it the next
colour, one the next shape. The door takes the avatar only when the key's glyph is the
door's; the level is done when every door has been entered. A bar counts the steps left
(what a move spends differs by level); a refill fills it again; when it runs out a life is
lost and the level starts over. A wall with a line along one edge throws the avatar that
enters the cell in front of that edge away from it, to the last cell before a wall. On
later levels some tiles slide to and fro along tracks.

From the frames alone:

* the avatar is the square drawn as two bands of colour (its size is the cell); the floor
  is what lies round it, walls are cells of one other colour;
* items are cells of floor with a small icon in the middle; doors are cells of another
  colour with a glyph in the middle; a wall cell with a line of a third colour along one
  edge makes the floor cell that edge faces a pusher;
* the key is the glyph drawn at twice the size in a box of background (read once a level);
  the steps left are the length of the bar along the bottom;
* what a tile does is learned when the avatar steps on it, from the key before and after:
  a quarter turn, a colour change (which colour follows which), a new shape (which follows
  which, carried across turns); a refill fills the bar; a pusher's landing cell is learned
  when it throws the avatar;
* a tile seen in another cell moves: its cells are kept move by move until they repeat,
  and the moves before that are spent on safe steps to watch it;
* a level in fog is one whose colours but one all lie within a disc round the avatar (the
  bottom rows aside): a cell is read only when the whole of it is in sight, what was seen is
  kept, and a tile no longer where it was seen is a mover, its cycle taken from the moves it
  was seen at.

The plan: an A* search over the avatar's cell, the key, the refills taken, the steps left
and the phase of the moving tiles, for the fewest moves into every door with the right key.
The estimate is the most of the walk to the door, the tile steps the key needs, and the
walk by way of any tile that every way of making the door's glyph needs; doors left are
chained in the cheapest order. While some door's glyph is out of reach of the tiles known,
the plan goes into a door that is not; when none is, the nearest step whose outcome is not
known is taken to see it. In fog, cells not seen are not walked on, and when there is no
plan the avatar walks to the nearest cell from which cells not seen yet come into view
(crossing a tile only at a toll), filling up at a refill first when the steps left would not
last the way there and on to one.
"""

from __future__ import annotations

import heapq
from collections import Counter, deque
from itertools import permutations

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
MAX_STATES = 1_000_000
MAX_LEVEL_ACTIONS = 400
MAX_FAILS = 8
FOG_RADIUS = 32  # a level whose colours but one all lie this near the avatar is in fog
HUD_ROWS = 9  # the bottom rows, where the counters are drawn over the fog
FOG_EXPLORE_STATES = 100_000  # a search for a tile's unknown effect, in fog, is cut here
MAX_APPS = 24  # tile steps looked through for turning a key into a door's glyph
MAX_PERIOD = 24  # the longest cycle looked for in a moving tile's sightings
TILE_TOLL = 6  # what walking over a tile costs an exploring walk in fog, in moves


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


Glyph = tuple  # a 3x3 glyph as a tuple of 9 colours, row by row


TURN = (6, 3, 0, 7, 4, 1, 8, 5, 2)  # a 3x3 glyph a quarter turn clockwise


def rot(g: Glyph, k: int = 1) -> Glyph:
    for _ in range(k % 4):  # k quarter turns clockwise
        g = tuple(g[i] for i in TURN)
    return g


def mask(g: Glyph, bg: int) -> tuple:
    return tuple(int(v != bg) for v in g)


def colours(g: Glyph, bg: int) -> set:
    return {v for v in g if v != bg}


# --- reading the screen ------------------------------------------------------------


def find_avatar(grid: np.ndarray, look: tuple | None = None):
    """(x, y, size, (top colour, bottom colour, top rows)) of the square drawn as two bands."""
    H, W = grid.shape
    out = []
    for s in (4, 5, 6, 7, 8):
        for y in range(0, H - s + 1):
            row = grid[y]
            for x in range(0, W - s + 1):
                a = int(row[x])
                if look is not None and a != look[0]:
                    continue
                if x > 0 and grid[y, x - 1] == a and (y == 0 or grid[y - 1, x] != a):
                    continue
                blk = grid[y:y + s, x:x + s]
                top = None
                for t in range(1, s):
                    if (blk[:t] == a).all() and len(np.unique(blk[t:])) == 1 and blk[t, 0] != a:
                        top = t
                if top is None:
                    continue
                b = int(blk[top, 0])
                if look is not None and (b, top, s) != (look[1], look[2], look[3]):
                    continue
                # the square stands alone: its colours stop at its edges
                edge = []
                if y > 0:
                    edge.append(grid[y - 1, x:x + s])
                if y + s < H:
                    edge.append(grid[y + s, x:x + s])
                if x > 0:
                    edge.append(grid[y:y + s, x - 1])
                if x + s < W:
                    edge.append(grid[y:y + s, x + s])
                if any(((e == a) | (e == b)).any() for e in edge):
                    continue
                out.append((x, y, s, (a, b, top, s)))
    return out[0] if len(out) == 1 else None


def find_key(grid: np.ndarray, bg: int):
    """The 3x3 glyph drawn at twice the size inside a ring of background: (x, y, glyph)."""
    H, W = grid.shape
    found = []
    for y in range(1, H - 6):
        for x in range(1, W - 6):
            win = grid[y:y + 6, x:x + 6]
            if (win == bg).all():
                continue
            ring = np.concatenate([grid[y - 1, x - 1:x + 7], grid[y + 6, x - 1:x + 7], grid[y:y + 6, x - 1], grid[y:y + 6, x + 6]])
            if not (ring == bg).all():
                continue
            small = win[::2, ::2]
            if not (np.repeat(np.repeat(small, 2, 0), 2, 1) == win).all():
                continue
            g = tuple(int(v) for v in small.ravel())
            if len(set(g)) < 2:
                continue
            found.append((x, y, g))
    return found[0] if len(found) == 1 else None


class Board:
    """The maze cut into cells: walls, floor, items (icons), doors (glyphs), pushers."""

    def __init__(self, grid: np.ndarray, x0: int, y0: int, s: int, floor: int) -> None:
        self.s, self.floor = s, floor
        self.ox, self.oy = x0 % s, y0 % s
        self.floor_cells: set = set()
        self.items: dict = {}  # cell -> icon (3x3 glyph in the middle)
        self.doors: dict = {}  # cell -> glyph
        self.pushers: dict = {}  # cell -> the edge its line is on
        self.cells: dict = {}
        H, W = grid.shape
        for j in range(-1, H // s + 1):
            for i in range(-1, W // s + 1):
                x, y = self.ox + i * s, self.oy + j * s
                if x < 0 or y < 0 or x + s > W or y + s > H:
                    continue
                self.cells[(i, j)] = grid[y:y + s, x:x + s]

    def read(self, grid: np.ndarray, avatar_cell, not_lines: set) -> None:
        s, f = self.s, self.floor
        m = s // 2
        self.not_lines = set(not_lines)
        # doors first: their frames' inside colour draws no pusher lines
        for c, p in self.cells.items():
            ring = np.concatenate([p[0], p[-1], p[1:-1, 0], p[1:-1, -1]])
            r0 = int(ring[0])
            if r0 != f and (ring == r0).all() and not (p[m - 1:m + 2, m - 1:m + 2] == r0).all():
                self.not_lines.add(r0)
        for c, p in self.cells.items():
            if c == avatar_cell:
                self.floor_cells.add(c)
                continue
            if (p == f).all():
                self.floor_cells.add(c)
                continue
            # floor with a strip of a door's frame along it
            vals = set(np.unique(p).tolist())
            if f in vals and vals - {f} <= self.not_lines and (p == f).sum() * 2 > p.size:
                self.floor_cells.add(c)
                continue
            ring = np.concatenate([p[0], p[-1], p[1:-1, 0], p[1:-1, -1]])
            centre = p[m - 1:m + 2, m - 1:m + 2]
            inner = p.copy()
            inner[m - 1:m + 2, m - 1:m + 2] = f
            if (ring == f).all() and (inner == f).all() and not (centre == f).all():
                self.items[c] = tuple(int(v) for v in centre.ravel())
                continue
            r0 = int(ring[0])
            if r0 != f and (ring == r0).all() and not (centre == r0).all():
                inner = p.copy()
                inner[m - 1:m + 2, m - 1:m + 2] = r0
                if (inner == r0).all():
                    self.doors[c] = tuple(int(v) for v in centre.ravel())
                    continue
            # a line along one edge of a wall: the floor cell it faces throws the avatar
            # away from it
            for d, rows, rest in (((0, -1), p[0], p[1:]), ((0, 1), p[-1], p[:-1]),
                                  ((-1, 0), p[:, 0], p[:, 1:]), ((1, 0), p[:, -1], p[:, :-1])):
                k = int(rows[0])
                w = int(rest.flat[0])
                if k in (f, w) or not (rows == k).all() or k in self.not_lines or not (rest == w).all():
                    continue
                self.pushers[add(c, d)] = d
                break

    def point(self, c):
        return self.ox + c[0] * self.s, self.oy + c[1] * self.s


def read_floor(grid: np.ndarray, x: int, y: int, s: int):
    """The floor: the colour of the cells round the avatar, the screen edges' colours last."""
    near: Counter = Counter()
    for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
        x0, y0 = x + dx, y + dy
        if x0 < 0 or y0 < 0 or x0 + s > grid.shape[1] or y0 + s > grid.shape[0]:
            continue
        p = grid[y0:y0 + s, x0:x0 + s]
        if len(np.unique(p)) == 1:
            near[int(p.flat[0])] += 1
    if not near:
        return None
    edge = {int(v) for v in grid[0]} | {int(v) for v in grid[:, 0]} | {int(v) for v in grid[:, -1]}
    return max(near, key=lambda c: (c not in edge, near[c]))


def find_bar(grid: np.ndarray, skip: set):
    """The steps bar: the longest run of one colour in the bottom rows (row, colour, x1, x2)."""
    best = None
    H, W = grid.shape
    for y in range(H - 8, H):
        row = grid[y]
        x = 0
        while x < W:
            c = int(row[x])
            x2 = x
            while x2 + 1 < W and row[x2 + 1] == c:
                x2 += 1
            if c not in skip and x2 - x + 1 >= 10 and (best is None or x2 - x > best[3] - best[2]):
                best = (y, c, x, x2)
            x = x2 + 1
    return best


def fog_of(grid: np.ndarray, x: int, y: int, s: int):
    """(fog colour, radius) when every colour but one lies within a disc round the avatar (the
    bottom rows, where counters are drawn, aside); None when the level is in plain view."""
    top = grid[:grid.shape[0] - HUD_ROWS]
    f = Counter(top.ravel().tolist()).most_common(1)[0][0]
    ys, xs = np.nonzero(top != f)
    if not len(xs):
        return None
    cx, cy = x + (s - 1) / 2, y + (s - 1) / 2
    r = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    return (int(f), r) if r < FOG_RADIUS else None


def locate(grid: np.ndarray, look: tuple):
    """Where the two-band square of this look is: (x, y), or None."""
    a, b, top, s = look
    H, W = grid.shape
    ys, xs = np.nonzero(grid == a)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if y + s > H or x + s > W or (y > 0 and grid[y - 1, x] == a) or (x > 0 and grid[y, x - 1] == a):
            continue
        blk = grid[y:y + s, x:x + s]
        if (blk[:top] == a).all() and (blk[top:] == b).all():
            return x, y
    return None


def key_colour(g: Glyph, bg: int):
    cs = colours(g, bg)
    return next(iter(cs)) if len(cs) == 1 else None


def recolour(g: Glyph, a: int, b: int) -> Glyph:
    return tuple(b if v == a else v for v in g)


# --- the solver --------------------------------------------------------------------


class KeySolver:
    name = "keys"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.look: tuple | None = None
        # what each tile (by its icon) does to the key, learned for the game
        self.effects: dict = {}
        self.applied: dict = {}  # (icon, key) -> the key after it, by the effects known
        self.cost: int | None = None  # steps a move spends
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.board: Board | None = None
        self.queue: list = []
        self.expect = None
        self.actions = 0
        self.fails = 0
        self.landing: dict = {}  # pusher cell -> where it throws the avatar (this level)
        self.stops: set = set()  # cells learned to stop the avatar
        self.cost_known = False  # a move's steps, seen again on each level
        self.fog = None  # (colour, radius) when the level is seen only round the avatar
        self.known: set = set()  # cells seen whole (under fog)
        self.seen_version = 0  # bumped when a door, an item or a pusher comes into view
        self.sighted: dict = {}  # item cell -> the moves (ticks) it was seen there (fog)
        self.searched_version = -1  # the map a search last found nothing on

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    # --- reading ---------------------------------------------------------------------

    def _setup(self, grid: np.ndarray) -> bool:
        av = find_avatar(grid, self.look)
        if av is None:
            return False
        x, y, s, look = av
        floor = read_floor(grid, x, y, s)
        if floor is None or floor in look[:2]:
            return False
        b = Board(grid, x, y, s, floor)
        cell = ((x - b.ox) // s, (y - b.oy) // s)
        b.read(grid, cell, set())
        self.grid_h = grid.shape[0]
        self._cells_on_screen = set(b.cells)
        self.fog = fog_of(grid, x, y, s)
        if self.fog is not None:
            # in fog only what lies wholly in sight is read; the doors may be anywhere
            self.board = b
            sight = self._view(cell)
            b.cells = {c: p for c, p in b.cells.items() if c in sight or c == cell}
            b.floor_cells = {c for c in b.floor_cells if c in b.cells}
            b.items = {c: i for c, i in b.items.items() if c in b.cells and c != cell}
            b.doors = {c: g for c, g in b.doors.items() if c in b.cells}
            b.pushers = {t: d for t, d in b.pushers.items() if (t[0] - d[0], t[1] - d[1]) in b.cells}
            self.known = set(b.cells)
            self.stats["fog_levels"] += 1
        elif not b.doors:
            return False
        key = None
        for bg in sorted(b.not_lines) + [c for c, _ in Counter(int(v) for v in grid.ravel()).most_common()]:
            if bg in (floor,) + look[:2]:
                continue
            key = find_key(grid, bg)
            if key is not None:
                self.bg = bg
                self.applied = {}
                break
        if key is None:
            return False
        uniform = Counter(int(p.flat[0]) for p in b.cells.values() if len(np.unique(p)) == 1)
        fog = self.fog[0] if self.fog is not None else None
        self.wall = next((c for c, _ in uniform.most_common() if c not in (floor, fog)), None)
        bar = find_bar(grid, {floor, self.bg, self.wall, fog} | set(look[:2]))
        if bar is None:
            return False
        self.board, self.cell, self.start_cell = b, cell, cell
        self.found_look = look
        self.key_at = key[:2]
        self.key = self.start_key = key[2]
        self.bar = bar
        self.full = bar[3] - bar[2] + 1
        self.pips = self._pips(grid)
        self.used: set = set()
        self.left: set = set(b.doors)
        self.static: dict = dict(b.items)  # items that have not moved
        self.movers: dict = {}  # icon -> {"track": cells by tick, "period": n or None}
        self.tick = 0  # moves made since the level (re)started
        # an item drawn in the bar's colour is taken for a refill
        for icon in set(b.items.values()):
            if icon not in self.effects and colours(icon, floor) == {bar[1]}:
                self.effects[icon] = ("refill",)
        return True

    def _view(self, c) -> set:
        """Under fog: the cells wholly in sight of the avatar standing in cell c."""
        b = self.board
        if getattr(self, "_sight", None) is None or self._sight[0] != (self.fog, b.s):
            s, (_, r) = b.s, self.fog
            reach = int(r // s) + 2
            offsets = set()
            for dj in range(-reach, reach + 1):
                for di in range(-reach, reach + 1):
                    # the cell's farthest corner from the avatar's centre, a pixel and a half spare
                    far = max(np.hypot(di * s + cx - (s - 1) / 2, dj * s + cy - (s - 1) / 2)
                              for cx in (0, s - 1) for cy in (0, s - 1))
                    if far <= r - 1.5:
                        offsets.add((di, dj))
            self._sight = ((self.fog, b.s), offsets)
        H = self.grid_h
        out = set()
        for o in self._sight[1]:
            n = add(c, o)
            y = b.oy + n[1] * b.s
            if n in self._cells_on_screen and y + b.s <= H - HUD_ROWS:
                out.add(n)
        return out

    def _merge(self, grid: np.ndarray, avatar) -> bool:
        """Under fog: the cells in sight now read and kept with the rest; True when a door, an
        item or a pusher came into view."""
        b = self.board
        fresh = Board(grid, b.ox, b.oy, b.s, b.floor)
        fresh.read(grid, avatar, set(b.not_lines))
        sight = self._view(avatar)
        new = False
        # an item no longer where it was seen (and not used up) moves: its sightings are its
        # first steps, and the icon seen elsewhere is it again
        for c0, icon in list(self.static.items()):
            if c0 in sight and c0 != avatar and c0 not in self.used and fresh.items.get(c0) != icon:
                del self.static[c0]
                m = self.movers.setdefault(icon, {"track": [], "period": None, "seen": {}})
                m.setdefault("seen", {})
                for t in self.sighted.pop(c0, []):
                    m["seen"][t] = c0
                self.stats["movers_seen"] += 1
        for icon, m in self.movers.items():
            if "seen" not in m:
                continue
            at = [c for c in sight if fresh.items.get(c) == icon and c != avatar]
            if len(at) == 1:
                m["seen"][self.tick] = at[0]
                self._cycle(m)
        for c in sight:
            if c not in fresh.cells or c == avatar:
                continue  # the avatar's own cell is read before it steps in
            b.cells[c] = fresh.cells[c]
            b.floor_cells.discard(c)
            b.items.pop(c, None)
            if c in fresh.floor_cells:
                b.floor_cells.add(c)
            elif c in fresh.items:
                icon = fresh.items[c]
                b.items[c] = icon
                if icon in self.movers:
                    pass  # a moving tile, seen where it is now
                elif c not in self.static:
                    self.static[c] = icon
                    new = True
                    if icon not in self.effects and colours(icon, b.floor) == {self.bar[1]}:
                        self.effects[icon] = ("refill",)
                if c in self.static:
                    self.sighted.setdefault(c, []).append(self.tick)
            elif c in fresh.doors:
                if c not in b.doors:
                    new = True
                    self.left.add(c)
                b.doors[c] = fresh.doors[c]
            self.known.add(c)
        for t, d in fresh.pushers.items():
            if (t[0] - d[0], t[1] - d[1]) in sight and t not in b.pushers:
                b.pushers[t] = d
                new = True
        b.not_lines |= fresh.not_lines
        if avatar not in self.known:
            # thrown into cells not seen yet: where the avatar stands is floor
            self.known.add(avatar)
            b.floor_cells.add(avatar)
            b.cells[avatar] = fresh.cells.get(avatar)
        if new:
            self.seen_version += 1
            self.stats["fog_seen"] += 1
        return new

    def _mover_now(self, m):
        """Where a moving tile is at this move: by its cycle, else as last tracked or seen."""
        tr, per = m["track"], m["period"]
        if per:
            return tr[self.tick % per]
        if tr and "seen" not in m:
            return tr[-1]
        return m.get("seen", {}).get(self.tick)

    def _cycle(self, m) -> None:
        """A moving tile's cycle from where it was seen at which moves: the shortest period
        that every sighting agrees with, every phase of it seen and some seen twice."""
        seen = m["seen"]
        if m["period"] is not None or len(seen) < 4:
            return
        for per in range(2, MAX_PERIOD + 1):
            phase: dict = {}
            if all(phase.setdefault(t % per, c) == c for t, c in seen.items()) and len(phase) == per and len(seen) >= per + 2:
                m["period"] = per
                m["track"] = [phase[r] for r in range(per)]
                self.stats["cycles_learned"] += 1
                return

    def _floor_paths(self, start, ends=()) -> tuple:
        """The cheapest walks over cells seen from start, pushers and doors kept off and a tile
        crossed only at a toll; cells in ends may be stepped onto but not walked through."""
        b = self.board
        dist, prev = {start: 0}, {start: None}
        heap = [(0, start)]
        while heap:
            g, c = heapq.heappop(heap)
            if g > dist[c] or (c in ends and c != start):
                continue
            for k, d in KEYS.items():
                n = add(c, d)
                if n in self.stops or n in b.pushers or n in b.doors:
                    continue
                if n in ends or n in b.floor_cells:
                    step = 1
                elif n in b.items:
                    step = 1 + TILE_TOLL
                else:
                    continue
                if g + step < dist.get(n, 1 << 30):
                    dist[n] = g + step
                    prev[n] = (c, k)
                    heapq.heappush(heap, (g + step, n))
        return dist, prev

    def _frontier(self):
        """Under fog: the fewest moves to a cell from which cells not seen yet come into view,
        by way of a refill first when the steps left would not last the walk there and on to a
        refill from it. (moves, expected states, True), or None."""
        b = self.board
        dec = self.cost or 1
        refills = {c for c, i in self.static.items()
                   if c not in self.used and self.effects.get(i, ("",))[0] == "refill" and c in b.items}
        dist, prev = self._floor_paths(self.cell, refills)
        gains = {c: len(self._view(c) - self.known) for c in dist if c not in refills}
        gains = {c: g for c, g in gains.items() if g}
        if not gains:
            return None
        target = min(gains, key=lambda c: (dist[c], -gains[c]))
        if refills:
            on, _ = self._floor_paths(target, refills)
            back = min((on[r] for r in refills if r in on), default=None)
            near = [r for r in refills if r in dist]
            if near and (back is None or self.pips < (dist[target] + back + 1) * dec):
                target = min(near, key=lambda r: dist[r])  # fill up on the way
        moves = []
        c = target
        while prev[c] is not None:
            p, k = prev[c]
            moves.append((k, c))
            c = p
        moves.reverse()
        out_moves, states = [], []
        st = (self.cell, self.key, frozenset(self.used), self.pips, frozenset(self.left))
        tick = self.tick
        for k, c in moves:
            r = self._move(st[0], c, st[1], st[2], st[3], st[4], tick) if st is not None else None
            st = r if isinstance(r, tuple) else None  # past a step the model cannot tell, expect nothing
            out_moves.append(k)
            states.append(st)
            tick += 1
        self.stats["fog_walks"] += 1
        return out_moves, states, True

    def _pips(self, grid: np.ndarray) -> int:
        y, c, x1, x2 = self.bar
        return int((grid[y, x1:x2 + 1] == c).sum())

    def _key_now(self, grid: np.ndarray) -> Glyph:
        x, y = self.key_at
        return tuple(int(v) for v in grid[y:y + 6:2, x:x + 6:2].ravel())

    def _cell_now(self, grid: np.ndarray):
        at = locate(grid, self.found_look)
        if at is None:
            return None
        b = self.board
        return ((at[0] - b.ox) // b.s, (at[1] - b.oy) // b.s)

    # --- the model -------------------------------------------------------------------

    def _apply(self, icon, key: Glyph):
        """The key after a tile with this icon, or None if not known."""
        if (icon, key) not in self.applied:
            self.applied[(icon, key)] = self._apply_now(icon, key)
        return self.applied[(icon, key)]

    def _apply_now(self, icon, key: Glyph):
        eff = self.effects.get(icon)
        if eff is None:
            return None
        kind = eff[0]
        if kind in ("none", "refill"):
            return key
        if kind == "turn":
            return rot(key, eff[1])
        kc = key_colour(key, self.bg)
        if kc is None:
            return None
        if kind == "colour":
            nc = eff[1].get(kc)
            return None if nc is None else recolour(key, kc, nc)
        # a new shape: known for this shape at some turn, in some colour
        norm = recolour(key, kc, -1)
        for k in range(4):
            r = eff[1].get(rot(norm, k))
            if r is not None:
                return recolour(rot(r, -k), -1, kc)
        return None

    def _slide(self, t, d):
        """Where a pusher at t is expected to throw the avatar: along d (away from its line)
        to the last cell before a wall or a door."""
        b = self.board
        c = t
        while True:
            n = add(c, d)
            if n in b.doors or not self._passable(n):
                return c
            c = n

    def _passable(self, t) -> bool:
        b = self.board
        if t in self.stops or t not in b.cells:
            return False
        return t in b.floor_cells or t in b.items or t in b.doors

    def _items_at(self, tick: int) -> dict:
        """Where the items are after this many moves (movers by their cycles)."""
        out = dict(self.static)
        for icon, m in self.movers.items():
            tr, per = m["track"], m["period"]
            if per:
                out[tr[tick % per]] = icon
            elif tick < len(tr):
                out[tr[tick]] = icon
        return out

    def _period(self) -> int:
        n = 1
        for m in self.movers.values():
            per = m["period"] or 1
            n = n * per // np.gcd(n, per)
        return n

    def _move(self, cell, t, key, used, pips, left, tick: int = 0, items=None):
        """The state after a move from cell into t: a tuple, None if it cannot be (or loses
        a life), or "explore" if what happens there is not known yet."""
        b = self.board
        if not self._passable(t):
            return None
        dec = self.cost or 1
        if t in b.doors and t in left:
            if key != b.doors[t]:
                return None
            left = left - {t}
        land = self.landing.get(t, t)
        if t in b.pushers and t not in self.landing:
            land = self._slide(t, b.pushers[t])
        refilled = False
        if items is None:
            items = self._items_at(tick + 1)
        if land in items and land not in used:
            icon = items[land]
            eff = self.effects.get(icon)
            if eff is None:
                return "explore"
            if eff[0] == "refill":
                used = used | {land}
                pips = self.full
                refilled = True
            else:
                key = self._apply(icon, key)
                if key is None:
                    return "explore"
        if not refilled:
            pips -= dec
            if pips < 0:
                return None
        return (land, key, used, pips, left)

    def _apps(self, key: Glyph, goal: Glyph, memo: dict, without=None) -> int:
        """The fewest tile steps that turn the key into the goal (by the effects known)."""
        if (key, goal, without) in memo:
            return memo[(key, goal, without)]
        icons = [i for i, e in self.effects.items() if e[0] in ("turn", "colour", "shape") and i != without]
        seen = {key: 0}
        q = deque([key])
        out = 99
        while q:
            g = q.popleft()
            if g == goal:
                out = seen[g]
                break
            if seen[g] >= MAX_APPS:
                continue
            for i in icons:
                n = self._apply(i, g)
                if n is not None and n not in seen:
                    seen[n] = seen[g] + 1
                    q.append(n)
        memo[(key, goal, without)] = out
        return out

    def _distances(self) -> dict:
        """Steps from every cell to every door, on the floor (pushers aside)."""
        b = self.board
        out = {}
        for door in b.doors:
            dist = {door: 0}
            q = deque([door])
            while q:
                c = q.popleft()
                for d in KEYS.values():
                    n = add(c, d)
                    if n not in dist and self._passable(n):
                        dist[n] = dist[c] + 1
                        q.append(n)
            out[door] = dist
        return out

    def _via(self, cells, dist: dict) -> dict:
        """Steps from every cell to the door by way of one of the cells."""
        out: dict = {}
        heap = [(dist[c], c) for c in cells if c in dist]
        heapq.heapify(heap)
        while heap:
            n, c = heapq.heappop(heap)
            if c in out:
                continue
            out[c] = n
            for d in KEYS.values():
                t = add(c, d)
                if t not in out and self._passable(t):
                    heapq.heappush(heap, (n + 1, t))
        return out

    def _search(self):
        """Moves into every door with the right key (or, while some door's glyph is out of
        reach of the tiles known, into one door that is not): (moves, expected states,
        exploring)."""
        period = self._period()
        at = [self._items_at(ph + 1) for ph in range(period)]
        start = (self.cell, self.key, frozenset(self.used), self.pips, frozenset(self.left), self.tick % period)
        prev = {start: None}
        best: dict = {}
        dists = self._distances()
        memo: dict = {}
        b = self.board

        tiles: dict = {}
        for ph in range(period):
            for c, i in self._items_at(ph).items():
                if self.effects.get(i, ("",))[0] in ("turn", "colour", "shape"):
                    tiles.setdefault(i, set()).add(c)
        via = {(i, d): self._via(cs, dists[d]) for i, cs in tiles.items() for d in b.doors}
        costs: dict = {}

        def cost(cell, key, d) -> int:
            """Moves at least, from the cell with the key, into door d."""
            if (cell, key, d) in costs:
                return costs[(cell, key, d)]
            out = dists[d].get(cell, 99)
            apps = self._apps(key, b.doors[d], memo)
            if apps < 99:
                out = max(out, apps)
                for i in tiles:
                    # a tile no transform can do without is on the way
                    if apps and self._apps(key, b.doors[d], memo, i) >= 99:
                        out = max(out, via[(i, d)].get(cell, 99))
            costs[(cell, key, d)] = out
            return out

        reach = {d for d in self.left if self._apps(self.key, b.doors[d], memo) < 99}
        if not reach:
            # no door's glyph is in reach of what is known: see a tile do something new
            return self._explore(start, period)
        whole = reach == self.left
        target = 0 if whole else len(self.left) - 1  # doors still shut when the plan ends
        hs: dict = {}

        def h(st) -> int:
            cell, key, _, _, left, _ = st
            if (cell, key, left) in hs:
                return hs[(cell, key, left)]
            if not left:
                out = 0
            elif not whole:
                out = min(cost(cell, key, d) for d in left & reach)
            elif len(left) > 3:
                out = min(cost(cell, key, d) for d in left)
            else:
                out = 1 << 30
                for order in permutations(left):
                    n = cost(cell, key, order[0])
                    for a, z in zip(order, order[1:]):
                        n += cost(a, b.doors[a], z)
                    out = min(out, n)
            hs[(cell, key, left)] = out
            return out

        heap = [(h(start), 0, 0, start)]
        tick = 0
        explore = None
        depth = {start: 0}

        def path(st, last=None):
            moves, states = [], []
            if last is not None:
                moves.append(last[0])
                states.append(last[1])
            while prev[st] is not None:
                p, k = prev[st]
                moves.append(k)
                states.append(st)
                st = p
            return moves[::-1], states[::-1]

        while heap:
            _, g, _, st = heapq.heappop(heap)
            if g > depth.get(st, 1 << 30):
                continue
            cell, key, used, pips, left, ph = st
            for k, d in KEYS.items():
                t = add(cell, d)
                r = self._move(cell, t, key, used, pips, left, ph, at[ph])
                if r is None:
                    continue
                if r == "explore":
                    if explore is None or g < explore[2]:
                        explore = (st, k, g)
                    continue
                r = r + ((ph + 1) % period,)
                if len(r[4]) <= target:
                    m, s = path(st, (k, r))
                    return m, s, False
                sig = (r[0], r[1], r[2], r[4], r[5])
                if best.get(sig, -1) >= r[3]:
                    continue
                best[sig] = r[3]
                if g + 1 >= depth.get(r, 1 << 30):
                    continue
                depth[r] = g + 1
                prev[r] = (st, k)
                tick += 1
                heapq.heappush(heap, (g + 1 + h(r), g + 1, tick, r))
            if len(prev) > MAX_STATES:
                self.stats["search_too_big"] += 1
                break
        if explore is not None:
            st, k, _ = explore
            m, s = path(st)
            return m + [k], s + [None], True
        return None

    def _explore(self, start, period: int):
        """The fewest moves to a step whose outcome is not known."""
        at = [self._items_at(ph + 1) for ph in range(period)]
        prev = {start: None}
        best: dict = {}
        q = deque([start])
        while q:
            st = q.popleft()
            cell, key, used, pips, left, ph = st
            for k, d in KEYS.items():
                r = self._move(cell, add(cell, d), key, used, pips, left, ph, at[ph])
                if r is None:
                    continue
                if r == "explore":
                    moves, states = [k], [None]
                    while prev[st] is not None:
                        p, kk = prev[st]
                        moves.append(kk)
                        states.append(st)
                        st = p
                    return moves[::-1], states[::-1], True
                r = r + ((ph + 1) % period,)
                sig = (r[0], r[1], r[2], r[4], r[5])
                if best.get(sig, -1) >= r[3]:
                    continue
                best[sig] = r[3]
                prev[r] = (st, k)
                q.append(r)
            if len(prev) > (MAX_STATES if self.fog is None else FOG_EXPLORE_STATES):
                break
        return None

    # --- learning from a step --------------------------------------------------------

    def _learn_item(self, icon, before: Glyph, after: Glyph, pips_before: int, pips_after: int, gone: bool) -> None:
        self.applied = {}
        if icon in self.effects and self.effects[icon][0] not in ("colour", "shape"):
            return
        if after == before:
            kind = ("refill",) if gone and pips_after >= pips_before else ("none",)
            self.effects.setdefault(icon, kind)
            self.stats[f"learned_{kind[0]}"] += 1
            return
        for k in (1, 3, 2):
            if rot(before, k) == after and colours(before, self.bg) == colours(after, self.bg):
                self.effects[icon] = ("turn", k)
                self.stats["learned_turn"] += 1
                return
        kb, ka = key_colour(before, self.bg), key_colour(after, self.bg)
        if kb is None or ka is None:
            return
        if mask(before, self.bg) == mask(after, self.bg):
            eff = self.effects.setdefault(icon, ("colour", {}))
            if eff[0] == "colour":
                eff[1][kb] = ka
                self.stats["learned_colour"] += 1
            return
        eff = self.effects.setdefault(icon, ("shape", {}))
        if eff[0] == "shape":
            eff[1][recolour(before, kb, -1)] = recolour(after, ka, -1)
            self.stats["learned_shape"] += 1

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
                if self.board is not None:
                    self.look = self.found_look
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        exp, self.expect = self.expect, None
        if action is None or self.board is None:
            return
        grid = np.asarray(obs.frame[-1])
        b = self.board
        if action is GameAction.RESET or exp is None:
            self.board = None
            self.read = False
            self.queue = []
            return
        before, t, predicted = exp
        cell = self._cell_now(grid)
        key = self._key_now(grid)
        pips = self._pips(grid)
        if cell is None:
            self._fail("avatar_lost")
            return
        _, key0, used0, pips0, left0 = before
        if cell == self.start_cell and key == self.start_key and pips == self.full and cell != t and (predicted is None or predicted[0] != cell):
            # a life lost: the level starts over
            self.stats["lives_lost"] += 1
            self.cell, self.key, self.pips = cell, key, pips
            self.used, self.left = set(), set(b.doors)
            self.tick = 0
            self.queue = []
            self._fail("reset")
            if self.fog is not None:
                self._merge(grid, cell)
            return
        if cell == before[0]:
            if t in b.doors and t in left0:
                self._fail("door_shut")
            else:
                self.stops.add(t)
                self._fail("stopped")
            self.key, self.pips = key, pips
            if self.fog is not None:
                self._merge(grid, cell)
            return
        if not self.cost_known and cell == t and t not in b.items and pips < pips0:
            self.cost = pips0 - pips
            self.cost_known = True
        if cell != t:
            self.landing[t] = cell
            self.stats["learned_push"] += 1
        if t in b.doors and t in self.left:
            self.left.discard(t)
        self.tick += 1
        if self.fog is not None:
            if self._merge(grid, cell):
                self.queue = []  # something came into view: plan again
        else:
            self._track(grid, cell)
        icon = None
        if cell in self.static and cell not in self.used:
            icon = self.static[cell]
        else:
            icon = next((i for i, m in self.movers.items() if self._mover_now(m) == cell), None)
        if icon is not None:
            self._learn_item(icon, key0, key, pips0, pips, pips == self.full and pips0 < self.full)
            if cell in self.static and (self.effects.get(icon, ("",))[0] == "refill" or pips == self.full and pips0 < self.full):
                self.used.add(cell)
        self.cell, self.key, self.pips = cell, key, pips
        if predicted is None or predicted[:2] != (cell, key) or predicted[3] != pips:
            if predicted is not None:
                self.stats["mismatch"] += 1
            self.queue = []

    def _track(self, grid: np.ndarray, avatar) -> None:
        """Items read again: one that left its cell moves, and its cells are kept by tick."""
        b = Board(grid, self.board.ox, self.board.oy, self.board.s, self.board.floor)
        b.read(grid, avatar, set(self.board.not_lines))
        now = {c: i for c, i in b.items.items() if c != avatar}
        for c, icon in list(self.static.items()):
            if c in self.used or c == avatar or now.get(c) == icon:
                continue
            new = [x for x, i in now.items() if i == icon and x not in self.static]
            if len(new) == 1 and icon not in self.movers:
                del self.static[c]
                self.movers[icon] = {"track": [c] * self.tick + [new[0]], "period": None}
                self.stats["movers_seen"] += 1
        for icon, m in self.movers.items():
            if len(m["track"]) > self.tick:
                continue
            new = [x for x, i in now.items() if i == icon and x not in self.static]
            m["track"].append(new[0] if len(new) == 1 else avatar)
            tr = m["track"]
            if m["period"] is None:
                for per in range(2, len(tr) - 1):
                    if all(tr[i] == tr[i - per] for i in range(per, len(tr))):
                        m["period"] = per
                        self.stats["cycles_learned"] += 1
                        break

    def _watch(self):
        """A safe move to see the movers go round: to a plain floor cell and back."""
        b = self.board
        if self.fog is not None and not any(m["period"] is None and m.get("seen") and max(m["seen"]) == self.tick
                                            for m in self.movers.values()):
            return []  # in fog a mover is watched only while it is in sight
        items = self._items_at(self.tick + 1)
        for k, d in KEYS.items():
            t = add(self.cell, d)
            if t in b.floor_cells and t not in b.pushers and t not in items and t not in b.doors and t not in self.stops:
                if self.pips - (self.cost or 1) > 2 * len(self.movers) + 4:
                    return [(k, None)]
        return []

    def _fail(self, why: str) -> None:
        self.fails += 1
        self.stats[why] += 1
        self.queue = []

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4} <= set(obs.available_actions):
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if self.board is None:
            if self.read:
                return None
            self.read = True
            if not self._setup(grid):
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
        if not self.queue and any(m["period"] is None for m in self.movers.values()):
            self.queue = self._watch()
            if self.queue:
                self.stats["watch_moves"] += 1
        if not self.queue:
            r = None
            if self.fog is None or (self.left and self.searched_version != self.seen_version):
                r = self._search()
                if r is None and self.fog is not None:
                    self.searched_version = self.seen_version  # not again until more is seen
            if r is None and self.fog is not None:
                r = self._frontier()
            if r is None:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            moves, states, exploring = r
            if exploring:
                self.stats["explore"] += 1
            self.queue = list(zip(moves, states))
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        k, predicted = self.queue.pop(0)
        before = (self.cell, self.key, frozenset(self.used), self.pips, frozenset(self.left))
        if predicted is not None:
            predicted = predicted[:5]
        self.expect = (before, add(self.cell, KEYS[k]), predicted)
        self.actions += 1
        self.stats["actions"] += 1
        return k, None
