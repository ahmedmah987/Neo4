"""The mirror family: move pieces and mirror lines until the pieces and their reflections
cover every target cell.

Some puzzles show a board of cells with pieces, mirror lines across the whole board, the
pieces' reflections in every mirror (and in the mirrors' reflections) drawn in a colour of
their own, and target cells. The keys move the selected object a cell, a mirror only across
its line; a click selects the piece under it, or else the mirror; nothing collides. A level
is done when every target cell is under a piece or a reflection.

From the frames alone:

* the lattice is the one on which every cell with something in it is drawn in one colour but
  for its middle pixel, where a mark may be drawn (on the selected object, a covered target);
* a mirror is a line of one colour across the board, under whatever else is drawn on it: a
  row or column in which no cell shows the background;
* the pieces are the cells of the colour whose reflections in the mirrors are exactly the
  cells of another colour, the images; the targets are the cells of the one colour left, and
  a target under a piece or an image shows that colour in its middle;
* the selected object has marks of a colour of its own; a mirror with no marks is fixed.

The colours' parts, once seen, are kept for the game's later levels, whose images may all lie
off the board at first.

The plan is every movable object's final place, the cheapest that covers every target: a key
for each cell an object moves, and a click to select each object moved but the one selected.
Nothing collides, so the order of the moves does not matter. After each action the board is
checked against the model, and read again if it differs.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {(0, -1): GameAction.ACTION1, (0, 1): GameAction.ACTION2, (-1, 0): GameAction.ACTION3, (1, 0): GameAction.ACTION4}
CLICK = GameAction.ACTION6
MAX_DEPTH = 13  # reflections of reflections, as deep as such a game draws them
MAX_MISMATCHES = 3
MAX_ACTIONS = 400
MAX_CONFIGS = 20_000  # mirror placements tried
MAX_PAIRS = 4_000_000  # sets of targets covered so far, times a piece's places


def find_lattice(grid: np.ndarray, bg: int):
    """(cell size, x offset, y offset, columns, rows): the lattice on which every cell with
    something drawn in it is one colour but for its middle pixel; None if there is none."""
    H, W = grid.shape
    best = None
    for p in range(2, 9):
        mid = (p // 2) * p + p // 2
        keep = [i for i in range(p * p) if i != mid]
        for oy in range(p):
            for ox in range(p):
                rows, cols = (H - oy) // p, (W - ox) // p
                if rows < 4 or cols < 4:
                    continue
                blk = grid[oy:oy + rows * p, ox:ox + cols * p].reshape(rows, p, cols, p)
                blk = blk.transpose(0, 2, 1, 3).reshape(rows, cols, p * p)[:, :, keep]
                busy = (blk != bg).any(axis=2)
                n = int(busy.sum())
                if n < 4:
                    continue
                score = float(((blk == blk[:, :, :1]).all(axis=2) & busy).sum()) / n
                if best is None or (score, -p) > best[0]:
                    best = ((score, -p), (p, ox, oy, cols, rows))
    if best is None or best[0][0] < 0.97:
        return None
    return best[1]


def read_cells(grid: np.ndarray, lat) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's colour (a corner pixel) and the colour in its middle."""
    p, ox, oy, cols, rows = lat
    blk = grid[oy:oy + rows * p, ox:ox + cols * p].reshape(rows, p, cols, p).transpose(0, 2, 1, 3)
    return blk[:, :, 0, 0].astype(int), blk[:, :, p // 2, p // 2].astype(int)


def reflections(mirrors) -> set:
    """Every map (sx, ax, sy, ay), x -> sx*x + ax and y -> sy*y + ay, that reflections in the
    mirrors (kind "V" at column c, "H" at row c) and in their images make."""
    maps = {(1, 0, 1, 0)}
    frontier = list(maps)
    for _ in range(MAX_DEPTH):
        nxt = []
        for sx, ax, sy, ay in frontier:
            for kind, c in mirrors:
                m = (-sx, 2 * c - ax, sy, ay) if kind == "V" else (sx, ax, -sy, 2 * c - ay)
                if m not in maps:
                    maps.add(m)
                    nxt.append(m)
        if not nxt:
            break
        frontier = nxt
    return maps


def images(cells, mirrors, cols: int, rows: int) -> set:
    """The cells and all their reflections that lie on the board."""
    out = set()
    for sx, ax, sy, ay in reflections(mirrors):
        for x, y in cells:
            q = (sx * x + ax, sy * y + ay)
            if 0 <= q[0] < cols and 0 <= q[1] < rows:
                out.add(q)
    return out


def components(cells: set) -> list[set]:
    """The groups of cells that touch, at a side or a corner."""
    out, left = [], set(cells)
    while left:
        stack = [left.pop()]
        comp = set(stack)
        while stack:
            x, y = stack.pop()
            for n in ((x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                if n in left:
                    left.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


class Piece:
    def __init__(self, cells: set) -> None:
        self.x = min(c[0] for c in cells)
        self.y = min(c[1] for c in cells)
        self.shape = frozenset((c[0] - self.x, c[1] - self.y) for c in cells)
        self.w = 1 + max(c[0] for c in self.shape)
        self.h = 1 + max(c[1] for c in self.shape)

    def cells(self, at=None) -> set:
        x, y = (self.x, self.y) if at is None else at
        return {(x + a, y + b) for a, b in self.shape}


class Board:
    """The board as read: roles of the colours, mirrors, pieces, targets and the selection."""

    def __init__(self) -> None:
        self.lat = None
        self.roles: dict = {}
        self.mirrors: list = []  # [kind, coordinate, fixed]
        self.pieces: list[Piece] = []
        self.targets: set = set()
        self.selected = None  # ("M", i) or ("P", i)

    def mirror_list(self, coords=None):
        return [(m[0], m[1] if coords is None else coords[i]) for i, m in enumerate(self.mirrors)]

    def point(self, c) -> tuple[int, int]:
        p, ox, oy, _, _ = self.lat
        return ox + c[0] * p + p // 2, oy + c[1] * p + p // 2


def _line(kind: str, c: int, cols: int, rows: int) -> list:
    return [(c, y) for y in range(rows)] if kind == "V" else [(x, c) for x in range(cols)]


def _lines(fill: np.ndarray, bg: int, M=None):
    """(the lines' colour, [(kind, coordinate)]): the rows and columns in which no cell shows
    the background, each with a cell of the lines' colour away from the lines across it. The
    colour, unless given, is the one most of their cells show."""
    rows, cols = fill.shape
    R = [y for y in range(rows) if (fill[y] != bg).all()]
    C = [x for x in range(cols) if (fill[:, x] != bg).all()]
    if M is None:
        n = Counter(fill[R].ravel().tolist()) + Counter(fill[:, C].ravel().tolist())
        if not n:
            return None, []
        M = n.most_common(1)[0][0]
    out = [("H", y) for y in R if any(fill[y, x] == M and x not in C for x in range(cols))]
    out += [("V", x) for x in C if any(fill[y, x] == M and y not in R for y in range(rows))]
    return M, out


def read_board(grid: np.ndarray, roles: dict | None = None) -> Board | None:
    """The board, or None if the frame is not one of this family's (with the colours' parts
    known from an earlier level when ``roles`` is given)."""
    bg = int(np.bincount(grid.ravel()).argmax())
    lat = find_lattice(grid, bg)
    if lat is None:
        return None
    fill, mark = read_cells(grid, lat)
    rows, cols = fill.shape
    if roles is not None and roles["bg"] != bg:
        return None
    M, mirrors = _lines(fill, bg, roles["M"] if roles else None)
    if not mirrors:
        return None
    cells_of = {}
    for y in range(rows):
        for x in range(cols):
            cells_of.setdefault(int(fill[y, x]), set()).add((x, y))
    lines = [_line(k, c, cols, rows) for k, c in mirrors]
    if cells_of[M] - {q for line in lines for q in line}:
        return None  # the lines' colour off the lines
    if roles is None:
        others = [c for c in cells_of if c not in (bg, M)]
        found = []
        for P, I in itertools.permutations(others, 2):
            Pc, Ic = cells_of[P], cells_of[I]
            if images(Pc, mirrors, cols, rows) - Pc == Ic:
                found.append((P, I))
        # pieces and images are each other's reflections; the pieces carry marks (they can
        # be selected), the images are drawn plain
        marked = {c: sum(mark[y, x] != c for x, y in cells_of[c]) for c in others}
        found.sort(key=lambda pi: marked[pi[1]] - marked[pi[0]])
        if not found or len(found) > 1 and marked[found[0][1]] - marked[found[0][0]] == marked[found[1][1]] - marked[found[1][0]]:
            return None  # no reading, or two and nothing to tell them apart
        P, I = found[0]
        Ic = cells_of[I]
        tcols = {c for c in others if c not in (P, I)} | {int(mark[y, x]) for x, y in Ic} - {I}
        if len(tcols) != 1:
            return None
        T = tcols.pop()
        on_objects = [int(mark[y, x]) for x, y in cells_of[P] | cells_of[M]]
        sel = [c for c in on_objects if c not in (bg, T, M, P)]
        roles = {"bg": bg, "M": M, "P": P, "I": I, "T": T, "S": Counter(sel).most_common(1)[0][0] if sel else None}
    P, I, T, S = roles["P"], roles["I"], roles["T"], roles["S"]
    Pc, Ic = cells_of.get(P, set()), cells_of.get(I, set())
    if not Pc or images(Pc, mirrors, cols, rows) - Pc != Ic:
        return None
    if set(cells_of) - {bg, M, P, I, T}:
        return None  # something else drawn on the board
    b = Board()
    b.lat, b.roles = lat, roles
    b.targets = set(cells_of.get(T, set())) | {(x, y) for y in range(rows) for x in range(cols) if mark[y, x] == T}
    if not b.targets:
        return None
    b.pieces = [Piece(comp) for comp in sorted(components(Pc), key=lambda s: min(s))]
    # where two lines cross, either may be drawn on top: their own cells are those away from
    # the crossings
    own = [[q for q in line if not any(q in other for j, other in enumerate(lines) if j != i)]
           for i, line in enumerate(lines)]
    for (kind, c), cells in zip(mirrors, own):
        marks = {int(mark[y, x]) for x, y in cells if fill[y, x] == M}
        b.mirrors.append([kind, c, marks <= {M}])
    if S is not None:
        # a selected mirror's marks are drawn on the pieces lying across its line too
        for i, cells in enumerate(own):
            if any(mark[y, x] == S and fill[y, x] == M for x, y in cells):
                b.selected = ("M", i)
        if b.selected is None:
            for i, p in enumerate(b.pieces):
                if any(mark[y, x] == S for x, y in p.cells()):
                    b.selected = ("P", i)
    return b


def plan_places(b: Board):
    """(cost, mirror coordinates, piece places): the cheapest final places that cover every
    target, or None. A key per cell moved, and a click for each object moved but the selected."""
    _, _, _, cols, rows = b.lat
    targets = sorted(b.targets)
    if len(targets) > 63:
        return None
    tidx = np.full((rows, cols), -1, dtype=np.int64)
    for i, (x, y) in enumerate(targets):
        tidx[y, x] = i
    full = np.uint64((1 << len(targets)) - 1)
    click = {o: (0 if o == b.selected else 1) for o in [("M", i) for i in range(len(b.mirrors))] + [("P", i) for i in range(len(b.pieces))]}
    opts = []  # each piece's places and the cost of reaching each
    for i, p in enumerate(b.pieces):
        xs, ys = np.meshgrid(np.arange(cols - p.w + 1), np.arange(rows - p.h + 1))
        xs, ys = xs.ravel(), ys.ravel()
        cost = np.abs(xs - p.x) + np.abs(ys - p.y)
        cost = cost + np.where(cost > 0, click[("P", i)], 0)
        shape = np.array(sorted(p.shape))
        opts.append((xs, ys, cost, shape))
    ranges = []
    for i, (kind, c, fixed) in enumerate(b.mirrors):
        n = cols if kind == "V" else rows
        ranges.append([c] if fixed else sorted(range(n), key=lambda v: abs(v - c)))
    configs = []
    for coords in itertools.product(*ranges):
        mc = sum(abs(v - m[1]) + (click[("M", i)] if v != m[1] else 0) for i, (v, m) in enumerate(zip(coords, b.mirrors)))
        configs.append((mc, coords))
    configs.sort()
    best = None
    for mc, coords in configs[:MAX_CONFIGS]:
        if best is not None and mc >= best[0]:
            break
        maps = reflections(b.mirror_list(coords))
        masks = []
        for xs, ys, cost, shape in opts:
            m = np.zeros(len(xs), dtype=np.uint64)
            for sx, ax, sy, ay in maps:
                X = sx * (xs[:, None] + shape[None, :, 0]) + ax
                Y = sy * (ys[:, None] + shape[None, :, 1]) + ay
                ok = (X >= 0) & (X < cols) & (Y >= 0) & (Y < rows)
                idx = np.where(ok, tidx[np.clip(Y, 0, rows - 1), np.clip(X, 0, cols - 1)], -1)
                bits = np.where(idx >= 0, np.left_shift(np.uint64(1), np.maximum(idx, 0).astype(np.uint64)), np.uint64(0))
                m |= np.bitwise_or.reduce(bits, axis=1)
            masks.append(m)
        # pieces one at a time: the cheapest way to each set of targets covered so far
        cur_m = np.array([0], dtype=np.uint64)
        cur_c = np.array([mc])
        back = []
        for (xs, ys, cost, _), m in zip(opts, masks):
            if len(cur_m) * len(m) > MAX_PAIRS:  # many pieces: only the cheapest sets so far
                keep = np.argsort(cur_c, kind="stable")[:MAX_PAIRS // len(m)]
                cur_m, cur_c = cur_m[keep], cur_c[keep]
                back[-1] = (back[-1][0][keep], back[-1][1][keep])
            om = (cur_m[:, None] | m[None, :]).ravel()
            oc = (cur_c[:, None] + cost[None, :]).ravel()
            order = np.lexsort((oc, om))
            om, oc = om[order], oc[order]
            first = np.ones(len(om), dtype=bool)
            first[1:] = om[1:] != om[:-1]
            keep = order[first]
            back.append((keep // len(m), keep % len(m)))
            cur_m, cur_c = om[first], oc[first]
        hit = np.nonzero(cur_m == full)[0]
        if len(hit) == 0:
            continue
        j = hit[np.argmin(cur_c[hit])]
        total = int(cur_c[j])
        if best is None or total < best[0]:
            places = []
            for (prev, k), (xs, ys, _, _) in zip(reversed(back), reversed(opts)):
                places.append((int(xs[k[j]]), int(ys[k[j]])))
                j = prev[j]
            best = (total, coords, list(reversed(places)))
    return best


class MirrorSolver:
    name = "mirror"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.roles: dict | None = None  # the colours' parts in this game, once seen
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.board: Board | None = None
        self.queue: list = []
        self.expect = None
        self.mismatches = 0
        self.actions = 0

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    # --- planning ----------------------------------------------------------------------

    def _plan(self) -> list:
        b = self.board
        found = plan_places(b)
        if found is None:
            return []
        _, coords, places = found
        moving = [("M", i) for i, (v, m) in enumerate(zip(coords, b.mirrors)) if v != m[1]]
        moving += [("P", i) for i, (q, p) in enumerate(zip(places, b.pieces)) if q != (p.x, p.y)]
        moving.sort(key=lambda o: o != b.selected)
        steps: list = []
        for o in moving:
            if o != b.selected:
                steps.append(("click", o))
            if o[0] == "M":
                kind, c, _ = b.mirrors[o[1]]
                d = coords[o[1]] - c
                step = (int(np.sign(d)), 0) if kind == "V" else (0, int(np.sign(d)))
                steps += [("key", o, step)] * abs(d)
            else:
                p, (x, y) = b.pieces[o[1]], places[o[1]]
                steps += [("key", o, (int(np.sign(x - p.x)), 0))] * abs(x - p.x)
                steps += [("key", o, (0, int(np.sign(y - p.y))))] * abs(y - p.y)
        return steps

    def _click_cell(self, o):
        """A cell whose click selects object o: one of its cells no other piece covers, and for
        a mirror none on a piece or on another mirror."""
        b = self.board
        _, _, _, cols, rows = b.lat
        if o[0] == "P":
            others = {c for j, q in enumerate(b.pieces) if j != o[1] for c in q.cells()}
            cands = [c for c in sorted(b.pieces[o[1]].cells()) if c not in others]
        else:
            kind, c, _ = b.mirrors[o[1]]
            on_pieces = {q for p in b.pieces for q in p.cells()}
            others = {q for j, (k, v, _) in enumerate(b.mirrors) if j != o[1] for q in _line(k, v, cols, rows)}
            cands = [q for q in _line(kind, c, cols, rows) if q not in on_pieces and q not in others]
        return cands[0] if cands else None

    # --- the solver interface ----------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty():
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
        if self.board is None or action is None or obs.state != GameState.NOT_FINISHED:
            return
        b = self.board
        if exp is not None:
            if exp[0] == "click":
                b.selected = exp[1]
            else:
                _, o, (dx, dy) = exp
                if o[0] == "M":
                    b.mirrors[o[1]][1] += dx if b.mirrors[o[1]][0] == "V" else dy
                else:
                    p = b.pieces[o[1]]
                    p.x, p.y = p.x + dx, p.y + dy
        seen = read_board(np.asarray(obs.frame[-1]), self.roles)
        if seen is not None and self._same(seen, b):
            return
        # the board is not what the model said (or another player moved): read it again
        self.stats["mismatch"] += 1
        self.mismatches += 1
        self.queue = []
        if seen is not None and len(seen.pieces) == len(b.pieces) and len(seen.mirrors) == len(b.mirrors):
            self.board = seen

    def _same(self, seen: Board, b: Board) -> bool:
        """The board read matches the model: the same mirror lines, the same piece cells."""
        if sorted((m[0], m[1]) for m in seen.mirrors) != sorted((m[0], m[1]) for m in b.mirrors):
            return False
        mine = set().union(*(p.cells() for p in b.pieces)) if b.pieces else set()
        theirs = set().union(*(p.cells() for p in seen.pieces)) if seen.pieces else set()
        if mine != theirs:
            return False
        # which piece is selected cannot always be told from the frame, only that one is
        return b.selected is None or seen.selected is None or seen.selected[0] == b.selected[0]

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4, 6} <= set(obs.available_actions):
            return None
        if self.mismatches > MAX_MISMATCHES or self.actions >= MAX_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        if self.board is None:
            if self.read:
                return None
            self.read = True
            b = read_board(np.asarray(obs.frame[-1]), self.roles)
            if b is None:
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
            self.board = b
            self.roles = b.roles
            self.stats["scenes"] += 1
        if not self.queue:
            self.queue = self._plan()
            if not self.queue:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            self.stats["plans"] += 1
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        step = self.queue.pop(0)
        self.expect = step
        self.actions += 1
        self.stats["actions"] += 1
        if step[0] == "click":
            cell = self._click_cell(step[1])
            if cell is None:
                self._give_up()
                return None
            x, y = self.board.point(cell)
            return CLICK, {"x": x, "y": y}
        return KEYS[step[2]], None
