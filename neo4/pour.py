"""The pour family: place the pieces so that the water poured from the spouts fills every cup.

Some puzzles show spouts with a drop of water under each, cups, a floor, and pieces: bars
that water runs along and falls off at both ends, and angles that turn it. A click picks a
piece, the keys move it a cell, and the fifth key pours: the water falls from the spouts,
spreads along whatever it lands on, turns where an angle sends it, and the level is done if
every cup is filled and no water touched the floor. A pour that fails is wiped and counted.

From the frames alone:

* the cell is the size of a drop; the way water falls is from a spout to its drop; the floor
  is the strip along the board's far end (and anything drawn like it);
* cups are U shapes (five cells round an opening), pieces the other objects: bars, angles
  (three cells of a square), and bars holding a spout of their own;
* the flow is simulated cell by cell: a drop goes on into empty cells, along a bar or a cup's
  rim it spreads to both sides, an angle turns it away from its arm, a cup's opening fills
  the cup, the board's edge takes it, the floor fails the pour;
* the piece picked out is drawn in a colour of its own; a key that does not move it shows a
  place it cannot go.

The plan: a search over where the pieces go, cheapest first (keys to move them, a click to
pick each one), led by the first thing that goes wrong in the simulated pour: a stream on
its way to the floor is caught by a piece put across it, a piece in the way is moved off.
A pour that does not do what was simulated is ruled out and the next plan tried.
"""

from __future__ import annotations

import heapq
from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
POUR = GameAction.ACTION5
CLICK = GameAction.ACTION6
MAX_POURS = 4
MAX_SIMS = 20000
MAX_FAILS = 6
MAX_STEPS = 400


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


# --- reading the screen ------------------------------------------------------------


def _cell_colour(grid: np.ndarray, x0: int, y0: int, s: int) -> int:
    """The colour of a cell, the status rows at the screen's top and bottom left out."""
    y1, y2 = max(y0, 1), min(y0 + s, 63)
    p = grid[y1:y2, x0:x0 + s] if y2 > y1 else grid[y0:y0 + s, x0:x0 + s]
    vals, counts = np.unique(p, return_counts=True)
    return int(vals[counts.argmax()])


def find_drops(grid: np.ndarray, bg: int):
    """(water colour, spout colour, cell size, gravity, drops) from the solid squares that sit
    against a cell of one other colour, all on one grid: the drops under the spouts."""
    found: dict = {}
    for c in fp.segment(grid):
        if c.color == bg or c.width != c.height or not 2 <= c.width <= 8 or c.area != c.width ** 2:
            continue
        s = c.width
        for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            x0, y0 = c.x1 - d[0] * s, c.y1 - d[1] * s
            if not (0 <= x0 and x0 + s <= 64 and 0 <= y0 and y0 + s <= 64):
                continue
            k = _cell_colour(grid, x0, y0, s)
            if k in (bg, c.color):
                continue
            # the drop's far side is open (it falls that way)
            x2, y2 = c.x1 + d[0] * s, c.y1 + d[1] * s
            if not (0 <= x2 and x2 + s <= 64 and 0 <= y2 and y2 + s <= 64) or _cell_colour(grid, x2, y2, s) != bg:
                continue
            found.setdefault((c.color, k, s, d), []).append((c.x1, c.y1))
    if not found:
        return None
    key = max(found, key=lambda k: len(found[k]))
    w, k, s, d = key
    drops = found[key]
    if len({(x % s, y % s) for x, y in drops}) != 1:
        return None
    return w, k, s, d, drops


def _components(cells: dict, colours: set) -> list[set]:
    seen, out = set(), []
    for c, k in cells.items():
        if k not in colours or c in seen:
            continue
        comp, q = set(), deque([c])
        seen.add(c)
        while q:
            a = q.popleft()
            comp.add(a)
            for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                b = add(a, d)
                if b not in seen and cells.get(b) == k:
                    seen.add(b)
                    q.append(b)
        out.append(comp)
    return out


def cup_opening(cells: set):
    """The opening cell and the way it faces, if these five cells are a U round it."""
    if len(cells) != 5:
        return None
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    box = {(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)}
    if len(box) != 6:
        return None
    gap = (box - cells).pop()
    if x2 - x1 == 2 and gap[0] == x1 + 1:
        return gap, (0, -1) if gap[1] == y1 else (0, 1)
    if y2 - y1 == 2 and gap[1] == y1 + 1:
        return gap, (-1, 0) if gap[0] == x1 else (1, 0)
    return None


class Piece:
    def __init__(self, cells: set, spouts: set) -> None:
        x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
        self.shape = frozenset((x - x0, y - y0) for x, y in cells)
        self.spouts = frozenset((x - x0, y - y0) for x, y in spouts)
        self.x, self.y = x0, y0
        self.kind = "angle" if len(cells) == 3 and max(a for a, _ in self.shape) == 1 and max(b for _, b in self.shape) == 1 else "bar"

    def cells(self, at=None) -> set:
        x, y = at if at is not None else (self.x, self.y)
        return {(x + a, y + b) for a, b in self.shape}


class Board:
    """The board read from a frame: what each cell holds, in screen cells."""

    def __init__(self) -> None:
        self.s = self.ox = self.oy = 0
        self.g = (0, 1)
        self.cells: dict = {}  # cell -> colour, for every cell on the board
        self.water: set = set()
        self.spouts: set = set()  # fixed spouts
        self.floor: set = set()
        self.walls: set = set()
        self.cups: list = []  # (cells, opening)
        self.pieces: list[Piece] = []

    def point(self, c) -> tuple[int, int]:
        return self.ox + c[0] * self.s + self.s // 2, self.oy + c[1] * self.s + self.s // 2


def read_board(grid: np.ndarray, known: tuple | None = None):
    """The board in a frame; known: (background, water, spout colour, cell size, gravity,
    origin) read from the level's first frame, where a drop hangs under every spout."""
    if known is None:
        counts = Counter(int(v) for v in grid[1:63].ravel())
        bg = counts.most_common(1)[0][0]
        r = find_drops(grid, bg)
        if r is None:
            return None
        w, k, s, d, drops = r
        ox, oy = drops[0][0] % s, drops[0][1] % s
    else:
        bg, w, k, s, d, (ox, oy) = known
    b = Board()
    b.s, b.g = s, d
    b.ox, b.oy = ox, oy
    for j in range(-1, 64 // s + 1):
        for i in range(-1, 64 // s + 1):
            x0, y0 = b.ox + i * s, b.oy + j * s
            if x0 >= 0 and y0 >= 0 and x0 + s <= 64 and y0 + s <= 64:
                b.cells[(i, j)] = _cell_colour(grid, x0, y0, s)
    b.water = {c for c, v in b.cells.items() if v == w}
    spout_cells = {c for c, v in b.cells.items() if v == k}
    # the floor: the colour along the board's far end
    far = [c for c in b.cells if all(add(c, (d[0] * n, d[1] * n)) not in b.cells for n in (1,))]
    fk = Counter(b.cells[c] for c in far).most_common(1)[0][0]
    if fk in (bg, w, k):
        fk = None
    if fk is not None:
        b.floor = {c for c, v in b.cells.items() if v == fk}
    used = {bg, w, k, fk}
    rest = {v for v in b.cells.values()} - used
    comps = _components(b.cells, rest)
    for comp in comps:
        op = cup_opening(comp)
        if op is not None:
            b.cups.append((frozenset(comp), op[0]))
    cup_cells = {c for comp, _ in b.cups for c in comp}
    others = [comp for comp in comps if not comp & cup_cells]
    # a spout between two parts of a bar belongs to it
    lone = set(spout_cells)
    groups: list = []
    for comp in others:
        groups.append([set(comp), set()])
    for sp in sorted(spout_cells):
        near = [g for g in groups if any(add(sp, dd) in g[0] for dd in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        if len(near) >= 1 and not any(add(sp, dd) in b.water for dd in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            merged = [set(), set()]
            for g in near:
                merged[0] |= g[0]
                merged[1] |= g[1]
                groups.remove(g)
            merged[0].add(sp)
            merged[1].add(sp)
            groups.append(merged)
            lone.discard(sp)
    b.spouts = lone
    for cells, sps in groups:
        if len(cells) > 12:
            b.walls |= cells
            continue
        b.pieces.append(Piece(cells, sps))
    b.bg, b.w, b.k, b.fk = bg, w, k, fk
    b.known = (bg, w, k, s, d, (ox, oy))
    return b


# --- the flow ----------------------------------------------------------------------


def _turn_of(g):
    """Screen from board: the turn that takes the board's down to the screen's g."""
    return {(0, 1): lambda v: v, (0, -1): lambda v: (-v[0], -v[1]),
            (1, 0): lambda v: (v[1], -v[0]), (-1, 0): lambda v: (-v[1], v[0])}[g]


def _sides(d, g):
    """The two sides of a drop moving d, in the order the flow tries them (fixed on the
    board, whichever way the screen shows it)."""
    m = _turn_of(g)
    inv = {(0, 1): (0, 1), (0, -1): (0, -1), (1, 0): (-1, 0), (-1, 0): (1, 0)}[g]
    mi = _turn_of(inv)
    db = mi(d)
    s1 = (-1, 0) if db[1] != 0 else (0, -1)
    return m(s1), m((-s1[0], -s1[1]))


def simulate(b: Board, places: list, limit: int = 4000):
    """Pour with the pieces at places: (cups filled, drops that reached the floor, cells
    wet)."""
    owner: dict = {}
    for i, p in enumerate(b.pieces):
        for c in p.cells(places[i]):
            owner.setdefault(c, ("piece", i))
    for n, (comp, _) in enumerate(b.cups):
        for c in comp:
            owner[c] = ("cup", n)
    for c in b.floor:
        owner[c] = ("floor", 0)
    for c in b.spouts | b.walls:
        owner[c] = ("stop", 0)
    wet = set(b.water)
    for c in wet:
        owner[c] = ("water", 0)
    g = b.g
    drops = [(c, g) for c in sorted(b.water)]
    for i, p in enumerate(b.pieces):
        x, y = places[i]
        for a, bb in sorted(p.spouts):
            t = (x + a + g[0], y + bb + g[1])
            if t in b.cells and t not in owner:
                owner[t] = ("water", 0)
                wet.add(t)
                drops.append((t, g))
    filled, floor = set(), 0
    steps = 0
    while drops and steps < limit:
        steps += 1
        nxt = []
        for c, d in drops:
            sides = _sides(d, g)
            n = add(c, d)
            if n not in b.cells:
                continue  # over the edge
            o = owner.get(n)
            if o is None:
                owner[n] = ("water", 0)
                wet.add(n)
                nxt.append((n, d))
                continue
            kind, i = o
            if kind == "water":
                nxt.append((n, d))
                continue

            def spread():
                for sd in sides:
                    t = add(c, sd)
                    if t in b.cells and t not in owner:
                        owner[t] = ("water", 0)
                        wet.add(t)
                        nxt.append((t, d))

            if kind == "piece" and b.pieces[i].kind == "bar":
                spread()
            elif kind == "cup":
                a1, a2 = owner.get(add(c, sides[0])), owner.get(add(c, sides[1]))
                if a1 == o and a2 == o:
                    filled.add(i)
                else:
                    spread()
            elif kind == "piece":
                a1, a2 = owner.get(add(c, sides[0])), owner.get(add(c, sides[1]))
                if a1 == o and add(c, sides[1]) not in owner:
                    nd = (d[1], -d[0])
                    t = add(c, nd)
                    if t in b.cells:
                        owner[t] = ("water", 0)
                        wet.add(t)
                        nxt.append((t, nd))
                elif a2 == o and add(c, sides[0]) not in owner:
                    nd = (-d[1], d[0])
                    t = add(c, nd)
                    if t in b.cells:
                        owner[t] = ("water", 0)
                        wet.add(t)
                        nxt.append((t, nd))
                else:
                    spread()
            elif kind == "floor":
                floor += 1
        drops = nxt
    return filled, floor, wet


# --- where pieces may go, and what it costs ------------------------------------------


class Rules:
    """What stops a piece, learned from keys that did not move it: a margin round the
    cups, the rows next to the spouts' edge, single places."""

    def __init__(self) -> None:
        self.cup_margin = False
        self.edge_rows = 0
        self.places: set = set()  # (piece shape, x, y) a piece did not go to


def edge_distance(b: Board, c) -> int:
    """How far a cell is from the board's edge the water comes from."""
    g = b.g
    n = 0
    t = c
    while True:
        t = (t[0] - g[0], t[1] - g[1])
        if t not in b.cells:
            return n
        n += 1


def allowed(b: Board, rules: Rules, p: Piece, at, fixed: set) -> bool:
    cells = p.cells(at)
    for c in cells:
        if c not in b.cells or c in fixed:
            return False
        if rules.edge_rows and edge_distance(b, c) < rules.edge_rows:
            return False
    if (p.shape, at[0], at[1]) in rules.places:
        return False
    if rules.cup_margin:
        for comp, _ in b.cups:
            for c in cells:
                if any(abs(c[0] - u[0]) <= 1 and abs(c[1] - u[1]) <= 1 for u in comp):
                    return False
    return True


def fixed_cells(b: Board) -> set:
    """Cells no piece may cover: everything drawn but the background and the pieces."""
    mine = {c for p in b.pieces for c in p.cells()}
    return {c for c, v in b.cells.items() if v != b.bg and c not in mine}


def reach(b: Board, rules: Rules, i: int, fixed: set) -> dict:
    """Keys from piece i's place to every place it can go: place -> list of keys."""
    p = b.pieces[i]
    start = (p.x, p.y)
    prev = {start: None}
    q = deque([start])
    while q:
        at = q.popleft()
        for k, d in KEYS.items():
            t = add(at, d)
            if t not in prev and allowed(b, rules, p, t, fixed):
                prev[t] = (at, k)
                q.append(t)
    out = {}
    for at in prev:
        keys, c = [], at
        while prev[c] is not None:
            c, k = prev[c]
            keys.append(k)
        out[at] = keys[::-1]
    return out


def outcome(b: Board, places: list):
    filled, floor, wet = simulate(b, places)
    win = len(filled) == len(b.cups) and not floor
    return win, len(filled), floor, wet


def plan(b: Board, rules: Rules, selected: int | None, ruled_out: set, beam: int = 30, depth: int = 6):
    """The cheapest placement found that the simulation says fills every cup: (places,
    moved pieces in order)."""
    fixed = fixed_cells(b)
    reaches = [reach(b, rules, i, fixed) for i in range(len(b.pieces))]
    start = tuple((p.x, p.y) for p in b.pieces)

    def cost(places) -> int:
        n = 0
        for i, at in enumerate(places):
            if at != start[i]:
                n += len(reaches[i][at]) + (0 if i == selected else 1)
        return n

    def score(places):
        win, filled, floor, wet = outcome(b, list(places))
        return win, filled, floor, wet

    def overlap(places, i, at) -> bool:
        mine = b.pieces[i].cells(at)
        return any(mine & b.pieces[j].cells(places[j]) for j in range(len(places)) if j != i)

    sims = 0
    seen = {start}
    win, filled, floor, wet = score(start)
    sims += 1
    if win and start not in ruled_out:
        return start
    layer = [(start, filled, floor, wet)]
    best = None
    for _ in range(depth):
        cands = []
        for places, _, _, wet in layer:
            for i, p in enumerate(b.pieces):
                touch = set()
                for at in reaches[i]:
                    cells = p.cells(at)
                    if cells & wet:
                        touch.add(at)
                if p.cells(places[i]) & wet:
                    # or out of the water's way
                    dry = sorted((a for a in reaches[i] if not p.cells(a) & wet), key=lambda a: len(reaches[i][a]))[:3]
                    touch.update(dry)
                for at in touch:
                    if at == places[i] or overlap(places, i, at):
                        continue
                    new = places[:i] + (at,) + places[i + 1:]
                    if new in seen:
                        continue
                    seen.add(new)
                    cands.append(new)
        scored = []
        found = best is not None
        for new in cands:
            if sims >= MAX_SIMS:
                break
            c = cost(new)
            if found and c >= best[0]:
                continue  # past a win, only cheaper ones count
            w, f, fl, wt = score(new)
            sims += 1
            if w and new not in ruled_out:
                if best is None or c < best[0]:
                    best = (c, new)
                continue
            scored.append((-(3 * f - fl), c, new, f, fl, wt))
        if found:
            # one change further than the first win, for a cheaper one
            return best[1]
        scored.sort(key=lambda t: (t[0], t[1]))
        layer = [(new, f, fl, wt) for _, _, new, f, fl, wt in scored[:beam]]
        if not layer:
            break
    return best[1] if best is not None else None


# --- the solver --------------------------------------------------------------------


class PourSolver:
    name = "pour"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.rules = Rules()
        self.sel_colour: int | None = None
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.board: Board | None = None
        self.known = None
        self.selected: int | None = None
        self.guessed = False
        self.queue: list = []
        self.expect: tuple | None = None
        self.ruled_out: set = set()
        self.pours = 0
        self.fails = 0
        self.actions = 0
        self.rules.places = set()

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    # --- reading ---------------------------------------------------------------------

    def _colours(self, grid: np.ndarray, i: int) -> Counter:
        b = self.board
        p = b.pieces[i]
        spouts = {(p.x + a, p.y + c) for a, c in p.spouts}
        out: Counter = Counter()
        for c in p.cells():
            if c not in spouts:
                x0, y0 = b.ox + c[0] * b.s, b.oy + c[1] * b.s
                out[_cell_colour(grid, x0, y0, b.s)] += 1
        return out

    def _find_selected(self, grid: np.ndarray) -> None:
        """The piece drawn in the picking colour; before that colour is known, a guess: the
        one piece drawn in a colour of its own (or the only piece), checked by the first key."""
        self.selected = None
        self.guessed = False
        if self.board is None:
            return
        cols = [self._colours(grid, i).most_common(1) for i in range(len(self.board.pieces))]
        cols = [c[0][0] if c else None for c in cols]
        if self.sel_colour is not None:
            if self.sel_colour in cols:
                self.selected = cols.index(self.sel_colour)
            return
        own = [i for i, c in enumerate(cols) if cols.count(c) == 1]
        if len(own) == 1:
            self.selected = own[0]
            self.guessed = True

    def _plan(self) -> list:
        b = self.board
        target = plan(b, self.rules, self.selected, self.ruled_out)
        if target is None:
            return []
        fixed = fixed_cells(b)
        order = sorted(range(len(b.pieces)), key=lambda i: i != self.selected)
        steps: list = []
        sel = self.selected
        for i in order:
            p = b.pieces[i]
            if target[i] == (p.x, p.y):
                continue
            keys = reach(b, self.rules, i, fixed).get(target[i])
            if keys is None:
                return []
            if sel != i:
                steps.append(("click", i))
                sel = i
            steps += [("key", i, k) for k in keys]
        steps.append(("pour", target))
        return steps

    # --- the solver interface --------------------------------------------------------

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
        exp, self.expect = self.expect, None
        if action is None or self.board is None:
            return
        grid = np.asarray(obs.frame[-1])
        b = self.board
        if action is GameAction.RESET or exp is None:
            # pieces may have moved without us: read them again
            nb = read_board(grid, self.known)
            if nb is not None and len(nb.pieces) == len(b.pieces):
                self.board = nb
                self._find_selected(grid)
            self.queue = []
            return
        kind = exp[0]
        if kind == "click":
            i = exp[1]
            self.selected = i
            col = self._colours(grid, i).most_common(1)
            if col:
                self.sel_colour = col[0][0]
            return
        if kind == "key":
            _, i, k = exp
            p = b.pieces[i]
            d = KEYS[k]
            moved = self._moved(grid, i, d)
            if moved:
                p.x, p.y = p.x + d[0], p.y + d[1]
                self.guessed = False
                col = self._colours(grid, i).most_common(1)
                if col:
                    self.sel_colour = col[0][0]
                return
            # another piece moved: that one was picked out
            for j in range(len(b.pieces)):
                if j != i and self._moved(grid, j, d):
                    q = b.pieces[j]
                    q.x, q.y = q.x + d[0], q.y + d[1]
                    self.selected = j
                    col = self._colours(grid, j).most_common(1)
                    if col:
                        self.sel_colour = col[0][0]
                    self.stats["selection_learned"] += 1
                    self.queue = []
                    return
            if self.selected != i or self.guessed:
                # it was not picked out after all
                self.queue = []
                self.selected = None
                self.guessed = False
                return
            self._learn_stop(i, (p.x + d[0], p.y + d[1]))
            self.queue = []
            return
        if kind == "pour":
            target = exp[1]
            self.pours += 1
            self.ruled_out.add(tuple(target))
            self.stats["pour_failed"] += 1
            frames = [np.asarray(f) for f in obs.frame]
            most = max(frames, key=lambda f: int((f == b.w).sum()))
            seen = {c for c in b.cells if _cell_colour(most, b.ox + c[0] * b.s, b.oy + c[1] * b.s, b.s) == b.w}
            _, _, wet = simulate(b, list(target))
            if seen != wet:
                self.stats["flow_mismatch"] += 1
                if not self.confirmed and len(seen & wet) < len(wet) // 2:
                    self.disabled = True  # the water does not go the way of this family
            self._find_selected(grid)
            self.queue = []

    def _moved(self, grid: np.ndarray, i: int, d) -> bool:
        """Piece i is now drawn a cell along d: its new cells drawn, the ones it left empty."""
        b = self.board
        p = b.pieces[i]
        old = p.cells()
        new = p.cells((p.x + d[0], p.y + d[1]))
        for c in new:
            if c not in b.cells:
                return False
            x0, y0 = b.ox + c[0] * b.s, b.oy + c[1] * b.s
            if _cell_colour(grid, x0, y0, b.s) == b.bg:
                return False
        others = {c for j, q in enumerate(b.pieces) if j != i for c in q.cells()}
        for c in old - new - others:
            x0, y0 = b.ox + c[0] * b.s, b.oy + c[1] * b.s
            if _cell_colour(grid, x0, y0, b.s) != b.bg:
                return False
        return True

    def _learn_stop(self, i: int, at) -> None:
        """A key did not move piece i to at: say why."""
        b = self.board
        p = b.pieces[i]
        cells = p.cells(at)
        near_cup = any(abs(c[0] - u[0]) <= 1 and abs(c[1] - u[1]) <= 1 for comp, _ in b.cups for u in comp for c in cells)
        rows = min((edge_distance(b, c) for c in cells if c in b.cells), default=99)
        if near_cup and not self.rules.cup_margin:
            self.rules.cup_margin = True
            self.stats["learned_cup_margin"] += 1
        elif rows < 4 and rows + 1 > self.rules.edge_rows:
            self.rules.edge_rows = rows + 1
            self.stats["learned_edge_rows"] += 1
        else:
            self.rules.places.add((p.shape, at[0], at[1]))
            self.stats["learned_place"] += 1
        self.fails += 1

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4, 5, 6} <= set(obs.available_actions):
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_STEPS or self.pours >= MAX_POURS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if self.board is None:
            if self.read:
                return None
            self.read = True
            b = read_board(grid)
            if b is None or not b.pieces or not b.cups:
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
            self.board = b
            self.known = b.known
            self._find_selected(grid)
        if not self.queue:
            self.queue = self._plan()
            if not self.queue:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        step = self.queue.pop(0)
        b = self.board
        if step[0] == "click":
            i = step[1]
            others = {c for j, q in enumerate(b.pieces) if j != i for c in q.cells()}
            cell = next((c for c in sorted(b.pieces[i].cells()) if c not in others), None)
            if cell is None:
                self._give_up()
                return None
            x, y = b.point(cell)
            self.expect = ("click", i)
            return self._do((CLICK, {"x": x, "y": y}))
        if step[0] == "key":
            self.expect = step
            return self._do((step[2], None))
        self.expect = step
        return self._do((POUR, None))

    def _do(self, a: Action) -> Action:
        self.actions += 1
        self.stats["actions"] += 1
        return a
