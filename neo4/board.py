"""The clue-board family: set the tiles of a board so that every clue is satisfied.

Some puzzles are a board of equal square blocks on a lattice. A *tile* is a block of
one colour, perhaps with marks of a second colour; a *clue* is a block whose centre
is a colour and whose ring is drawn in mark colours, one cell for each neighbour.
Clicking a tile turns it to the next colour of the palette, and a marked tile also
turns the neighbours its marks point at. A clue's mark says what the neighbour in its
direction must be: the clue's colour, or another one. With every clue satisfied the
level is done.

From the frames alone:

* a block is a square of 3x3 equal cells with a plain ring round it; one of one
  colour, or of one colour and marks, is a tile; one whose centre colour is nowhere
  in its ring is a clue;
* the palette is a row or column of small squares off the board, read in order (the
  tiles' colours when there is none), and what a click does to a tile's colour is
  checked on every click;
* what a mark of a clue means is not known in advance: the hypotheses are tried in
  order of the solved examples the level's first frame shows (clues already
  satisfied), then of cost; one that was played to the end without completing the
  level is ruled out, and the meanings that completed a level are kept;
* what a click turns is predicted from the tile's marks (the tile and the
  neighbours they point at) and learned from what it did; a tile that a click does
  not turn is no tile (a drawing of one).

The plan is the fewest clicks: tiles coupled by what their clicks turn are solved
together by a depth-first search, a tile's constraint checked as soon as every
click that turns it is chosen.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

SAME, OTHER = "same", "other"
CELL_SIZES = (2, 1, 3, 4)  # pixels per cell, tried in this order
MAX_SQUARE = 8  # a palette square is at most this many pixels a side
MAX_NODES = 200000  # search steps per plan
MAX_MARKS = 6  # mark colours whose meaning is searched at once
MAX_FAILS = 8  # clicks that did nothing, or a hypothesis played out, per level
MAX_LEVEL_ACTIONS = 300


@dataclass
class Tile:
    color: int
    marks: frozenset  # (dx, dy) lattice offsets the marks point at


@dataclass
class Clue:
    color: int
    ring: dict  # (dx, dy) lattice offset -> mark colour


@dataclass
class Board:
    cell: int  # pixels per cell
    pitch: int  # pixels from a block to the next
    tiles: dict = field(default_factory=dict)  # (x, y) top-left -> Tile
    clues: dict = field(default_factory=dict)  # (x, y) top-left -> Clue
    palette: list = field(default_factory=list)  # the shown palette, in order

    def at(self, p: tuple[int, int], d: tuple[int, int]) -> tuple[int, int]:
        return (p[0] + d[0] * self.pitch, p[1] + d[1] * self.pitch)

    def click_point(self, p: tuple[int, int]) -> tuple[int, int]:
        half = 3 * self.cell // 2
        return (p[0] + half, p[1] + half)


def find_blocks(grid: np.ndarray, s: int) -> dict:
    """Every square of 3x3 cells of s x s equal pixels whose one-pixel ring is of one
    colour found nowhere in it: (x, y) -> the 3x3 cell colours."""
    H, W = grid.shape
    n = 3 * s
    if H < n or W < n:
        return {}
    h, w = H - s + 1, W - s + 1
    uniform = np.ones((h, w), dtype=bool)
    base = grid[:h, :w]
    for dy in range(s):
        for dx in range(s):
            if dy or dx:
                uniform &= grid[dy:dy + h, dx:dx + w] == base
    hh, ww = H - n + 1, W - n + 1
    ok = np.ones((hh, ww), dtype=bool)
    for j in range(3):
        for i in range(3):
            ok &= uniform[j * s:j * s + hh, i * s:i * s + ww]
    out = {}
    for y, x in zip(*np.nonzero(ok)):
        y, x = int(y), int(x)
        cells = grid[y:y + n:s, x:x + n:s]
        ring = []
        x1, x2 = max(x - 1, 0), min(x + n + 1, W)
        if y > 0:
            ring.append(grid[y - 1, x1:x2])
        if y + n < H:
            ring.append(grid[y + n, x1:x2])
        if x > 0:
            ring.append(grid[y:y + n, x - 1])
        if x + n < W:
            ring.append(grid[y:y + n, x + n])
        if not ring:
            continue
        ring = np.concatenate(ring)
        r = ring[0]
        if (ring != r).any() or (cells == r).any():
            continue
        out[(x, y)] = cells.astype(int)
    return out


def _classify(cells: np.ndarray) -> Tile | Clue | None:
    centre = int(cells[1, 1])
    ring = {(i - 1, j - 1): int(cells[j, i]) for j in range(3) for i in range(3) if (i, j) != (1, 1)}
    colours = set(ring.values())
    if colours == {centre}:
        return Tile(centre, frozenset())
    if centre in colours:
        others = colours - {centre}
        if len(others) != 1:
            return None
        m = others.pop()
        return Tile(centre, frozenset(d for d, c in ring.items() if c == m))
    return Clue(centre, ring)


def _pitch(blocks: list[tuple[int, int]], n: int) -> int | None:
    """The commonest distance from a block to the nearest one right of it or below it."""
    steps: Counter = Counter()
    pos = set(blocks)
    for x, y in blocks:
        right = [bx - x for bx, by in pos if by == y and bx - x >= n]
        below = [by - y for bx, by in pos if bx == x and by - y >= n]
        for d in (right, below):
            if d:
                steps[min(d)] += 1
    if not steps:
        return None
    return steps.most_common(1)[0][0]


def _palette(grid: np.ndarray, covered: np.ndarray) -> list[list[int]]:
    """Rows and columns of equal small solid squares off the board, each read in
    order: candidate palettes."""
    squares = []
    for c in fp.segment(grid):
        if c.width != c.height or not 2 <= c.width <= MAX_SQUARE or not c.is_rectangle:
            continue
        if covered[c.y1:c.y1 + c.height, c.x1:c.x1 + c.width].any():
            continue
        squares.append((c.x1, c.y1, c.width, c.color))
    at = {(q[0], q[1], q[2]): q for q in squares}
    out = []
    used: set = set()
    for sq in sorted(squares, key=lambda q: (q[1], q[0])):
        if sq in used:
            continue
        w = sq[2]
        for dx, dy in ((0, w), (w, 0)):  # a column, else a row
            chain = [sq]
            while (chain[-1][0] + dx, chain[-1][1] + dy, w) in at:
                chain.append(at[(chain[-1][0] + dx, chain[-1][1] + dy, w)])
            if len(chain) >= 2:
                used.update(chain)
                cols: list[int] = []
                for q in chain:
                    if q[3] not in cols:
                        cols.append(q[3])
                out.append(cols)
                break
    return out


def read_board(grid: np.ndarray) -> Board | None:
    """The board the frame shows, or None."""
    grid = np.asarray(grid)
    for s in CELL_SIZES:
        blocks = find_blocks(grid, s)
        if len(blocks) < 3:
            continue
        n = 3 * s
        pitch = _pitch(list(blocks), n)
        if pitch is None:
            continue
        board = Board(s, pitch)
        for p, cells in blocks.items():
            kind = _classify(cells)
            if isinstance(kind, Tile):
                board.tiles[p] = kind
            elif isinstance(kind, Clue):
                board.clues[p] = kind
        covered = np.zeros(grid.shape, dtype=bool)
        for x, y in blocks:
            covered[max(y - 1, 0):y + n + 1, max(x - 1, 0):x + n + 1] = True
        # the palette shown: the row of squares holding the most tiles' colours
        best = 0
        for cols in _palette(grid, covered):
            held = sum(1 for t in board.tiles.values() if t.color in cols)
            if held > best and 2 * held >= len(board.tiles):
                board.palette, best = cols, held
        if board.palette:
            board.tiles = {p: t for p, t in board.tiles.items() if t.color in board.palette}
        tile_colours = {t.color for t in board.tiles.values()}
        # a clue's marks are no tile colour; a tile's colour is in the palette shown
        shown = set(board.palette) | tile_colours
        board.clues = {p: c for p, c in board.clues.items() if not set(c.ring.values()) & shown}
        if not board.clues:
            continue
        if not any(board.at(p, d) in board.tiles for p, c in board.clues.items() for d in c.ring):
            continue
        return board
    return None


def solve(variables: list, effects: dict, init: dict, allowed: dict, k: int,
          max_nodes: int = MAX_NODES) -> dict | None:
    """The fewest clicks (variable -> clicks, 0..k-1) after which every tile u with
    allowed[u] has a colour index (init[u] plus the clicks that turn it, mod k) in it;
    None if there is none (or the search ran out of steps without one)."""
    turned_by: dict = {}
    for t in variables:
        for u in effects[t]:
            turned_by.setdefault(u, []).append(t)
    for u in allowed:
        if not allowed[u]:
            return None
        if u not in turned_by and init[u] not in allowed[u]:
            return None
    # coupled groups
    parent = {t: t for t in variables}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, ts in turned_by.items():
        for t in ts[1:]:
            parent[find(t)] = find(ts[0])
    groups: dict = {}
    for t in variables:
        groups.setdefault(find(t), []).append(t)
    out: dict = {}
    nodes = [0]
    for members in groups.values():
        cons = [u for u in allowed if u in turned_by and find(turned_by[u][0]) == find(members[0])]
        if not cons:
            continue
        order = sorted(members, key=lambda t: (t[1], t[0]))
        index = {t: i for i, t in enumerate(order)}
        decided: list = [[] for _ in order]
        for u in cons:
            decided[max(index[t] for t in turned_by[u])].append(u)
        value = {u: init[u] for u in cons}
        cons_set = set(cons)
        best: list = [None, None]
        choice = [0] * len(order)

        def dfs(i: int, cost: int) -> None:
            if best[0] is not None and cost >= best[0]:
                return
            if nodes[0] > max_nodes:
                return
            nodes[0] += 1
            if i == len(order):
                best[0], best[1] = cost, list(choice)
                return
            t = order[i]
            hit = [u for u in effects[t] if u in cons_set]
            for v in range(k):
                if v:
                    for u in hit:
                        value[u] = (value[u] + 1) % k
                choice[i] = v
                if all(value[u] in allowed[u] for u in decided[i]):
                    dfs(i + 1, cost + v)
            for u in hit:
                value[u] = (value[u] - (k - 1)) % k  # undo the k-1 turns
            choice[i] = 0

        dfs(0, 0)
        if best[1] is None:
            return None
        for t, v in zip(order, best[1]):
            if v:
                out[t] = v
    return out


class ClueSolver:
    name = "clue"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.meaning: dict = {}  # mark colour -> SAME or OTHER, from completed levels
        self.turns: dict = {}  # a tile's marks -> lattice offsets its click turns
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.fails = 0
        self.actions = 0
        self.first: Board | None = None  # the level's first frame
        self.fixed: set = set()  # blocks a click did not turn
        self.ruled_out: list = []  # hypotheses played out without completing the level
        self.trial: dict | None = None  # the meanings being played
        self.nxt: dict = {}  # colour -> the colour a click turned it to, this level
        self.expect: tuple | None = None  # (tile clicked, colours before)

    # --- watching -----------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
                if self.trial:
                    self.meaning.update(self.trial)
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        if action is None or action is GameAction.RESET:
            self.expect = None
            return
        if self.expect is None:
            return
        clicked, before, board = self.expect
        self.expect = None
        if action is not GameAction.ACTION6 or not data or (data.get("x"), data.get("y")) != board.click_point(clicked):
            return
        after = read_board(np.asarray(obs.frame[-1]))
        if after is None:
            return
        changed = [p for p, c in before.items() if p in after.tiles and after.tiles[p].color != c]
        if not changed:
            self.fixed.add(clicked)
            self.fails += 1
            self.stats["fixed"] += 1
            if not self.confirmed:
                self._give_up()  # not shown to be a board game yet: no more blind clicks
            return
        for p in changed:
            self.nxt[before[p]] = after.tiles[p].color
        kind = board.tiles[clicked].marks
        offsets = {((p[0] - clicked[0]) // board.pitch, (p[1] - clicked[1]) // board.pitch) for p in changed}
        # an offset that has no tile could not be seen: keep what was predicted there
        unseen = {d for d in self._turns(kind) if board.at(clicked, d) not in board.tiles}
        learned = frozenset(offsets | unseen)
        if learned != self._turns(kind):
            self.stats["turns_learned"] += 1
            self.turns[kind] = learned

    def _turns(self, marks: frozenset) -> frozenset:
        return self.turns.get(marks, marks | {(0, 0)})

    # --- planning -----------------------------------------------------------------

    def _cycle(self, board: Board) -> list[int]:
        cycle = list(board.palette)
        if not cycle:
            for p in sorted(board.tiles, key=lambda q: (q[1], q[0])):
                if board.tiles[p].color not in cycle:
                    cycle.append(board.tiles[p].color)
            for c in board.clues.values():
                if c.color not in cycle:
                    cycle.append(c.color)
        # what clicks did overrides the order shown
        for a, b in self.nxt.items():
            if a not in cycle:
                continue
            if b in cycle:
                cycle.remove(b)
            cycle.insert(cycle.index(a) + 1, b)
        return cycle

    def _constraints(self, board: Board) -> list:
        tiles = {p for p in board.tiles if p not in self.fixed}
        out = []
        for p, c in board.clues.items():
            for d, m in c.ring.items():
                q = board.at(p, d)
                if q in tiles:
                    out.append((p, q, m, c.color))
        return out

    @staticmethod
    def _satisfied(board: Board, cons: list, meaning: dict) -> tuple[int, bool]:
        """How many clues the board satisfies under the meanings, and whether all do."""
        ok: dict = {}
        for p, q, m, colour in cons:
            if m not in meaning or q not in board.tiles:
                continue
            same = board.tiles[q].color == colour
            good = same if meaning[m] == SAME else not same
            ok[p] = ok.get(p, True) and good
        n = sum(ok.values())
        return n, n == len(ok)

    def _solve(self, board: Board, cons: list, meaning: dict, cycle: list[int]) -> dict | None:
        k = len(cycle)
        idx = {c: i for i, c in enumerate(cycle)}
        tiles = [p for p in board.tiles if p not in self.fixed and board.tiles[p].color in idx]
        tile_set = set(tiles)
        effects = {}
        for p in tiles:
            effects[p] = [q for q in (board.at(p, d) for d in self._turns(board.tiles[p].marks)) if q in tile_set]
        init = {p: idx[board.tiles[p].color] for p in tiles}
        allowed: dict = {}
        for _, q, m, colour in cons:
            if q not in tile_set:
                continue
            if meaning[m] == SAME:
                want = {idx[colour]} if colour in idx else set()
            else:
                want = set(range(k)) - ({idx[colour]} if colour in idx else set())
            allowed[q] = allowed.get(q, set(range(k))) & want
        return solve(tiles, effects, init, allowed, k)

    def _plan(self, board: Board) -> list | None:
        cons = self._constraints(board)
        marks = sorted({m for _, _, m, _ in cons})
        if not cons or len(marks) > MAX_MARKS:
            return None
        cycle = self._cycle(board)
        first = self.first or board
        first_cons = [c for c in self._constraints(first) if c[1] in first.tiles]
        options = []
        for combo in itertools.product((SAME, OTHER), repeat=len(marks)):
            meaning = dict(zip(marks, combo))
            if meaning in self.ruled_out:
                continue
            _, done = self._satisfied(board, cons, meaning)
            if done:
                continue  # the level would be over
            shown, _ = self._satisfied(first, first_cons, meaning) if first_cons else (0, False)
            known = sum(1 for m, v in meaning.items() if m in self.meaning and self.meaning[m] != v)
            options.append((known, -shown, combo, meaning))
        options.sort(key=lambda o: o[:2])
        best = None
        for known, shown, _, meaning in options:
            if best is not None and (known, shown) != best[0]:
                break
            clicks = self._solve(board, cons, meaning, cycle)
            if clicks is None:
                continue
            cost = sum(clicks.values())
            if best is None or cost < best[1]:
                best = ((known, shown), cost, clicks, meaning)
        if best is None:
            # nothing at the best rank: the next ranks in turn
            for known, shown, _, meaning in options:
                clicks = self._solve(board, cons, meaning, cycle)
                if clicks is not None:
                    best = ((known, shown), sum(clicks.values()), clicks, meaning)
                    break
        if best is None:
            return None
        _, _, clicks, meaning = best
        self.trial = meaning
        return sorted(clicks, key=lambda p: (p[1], p[0]))

    # --- acting -------------------------------------------------------------------

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.expect = None

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if 6 not in obs.available_actions:
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        board = read_board(np.asarray(obs.frame[-1]))
        if board is None:
            # not a board (looked at once a level), or no longer readable
            self.gave_up = not self.read or self.in_control
            self.in_control = False
            self.read = True
            return None
        if not self.read:
            self.first = board
        self.read = True
        for _ in range(MAX_FAILS):
            plan = self._plan(board)
            if plan is None:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            if plan:
                break
            # the meanings played out and the level goes on: they are wrong
            self.stats["ruled_out"] += 1
            self.ruled_out.append(self.trial)
            self.fails += 1
        else:
            self._give_up()
            return None
        if not self.in_control:
            self.in_control = True
            self.stats["takeovers"] += 1
        tile = plan[0]
        before = {p: t.color for p, t in board.tiles.items()}
        self.expect = (tile, before, board)
        self.actions += 1
        self.stats["clicks"] += 1
        x, y = board.click_point(tile)
        return GameAction.ACTION6, {"x": x, "y": y}
