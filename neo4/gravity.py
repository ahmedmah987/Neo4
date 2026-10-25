"""The gravity family: an avatar that walks left and right and falls with gravity, through a
tall shaft of tiles that clicks change, to a gem.

Some puzzles show a shaft of cells on a background: walls, tiles of several kinds, spikes and
a gem, the avatar resting against a wall. Two keys move the avatar a cell left or right, and
it falls, the way gravity pulls, until something stops it. A click on a tile changes it: one
kind is cleared, another turns from solid to open and back, another turns gravity round. A
fall that ends on spikes ends the go, and so does a bar of steps that fills. The view follows
the avatar up and down the shaft, so most of it is out of sight.

From the frames alone:

* the cells are a step of the avatar's apart (its first move shows it), on the lattice where
  the frame shows the fewest kinds of cell; a cell's look is what is drawn inside it;
* the avatar is the small shape in colours drawn nowhere else, the one the first move moves;
  later, its main colour's patch, as large as it was then;
* how far the view moved is summed over the frames of each action, so every cell seen keeps
  its place in a map of the shaft, and the map is kept when the level is begun again;
* a look drawn mostly in the background's colour is taken to be open, any other to be solid,
  until a move or a fall shows otherwise; what a look does when clicked is seen the first time
  one is clicked (a tile that went out of view as the avatar fell is taken to have vanished);
  a look the avatar has rested on is safe, one it died on is not;
* gravity is the way the avatar rests; a click that turns it shows as the avatar resting
  elsewhere on screen;
* the gem is the look that ended a level; before that, a look seen only once is tried.

The plan: a search, cheapest first, over the avatar's cell, gravity and the tiles changed, by
moves and clicks made just before the step that needs them (a tile on the avatar's way cleared,
the one it stands on cleared, one turned solid to stop a fall, gravity turned), for the gem, or
while it is not in the map, for a view of rows not yet seen, or else a fall on into them. A
landing on a look not yet rested on costs more, and a view is not taken from a place the
avatar could leave only by such a landing. With no plan, a look whose click has not been seen
is clicked once, and with none of those, the level is begun again.
"""

from __future__ import annotations

import heapq
from collections import Counter

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

LEFT, RIGHT, CLICK = GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION6
MAX_SHIFT = 12  # pixels the view can move between two frames
MAX_PRE = 4  # clicks made ready before the step that needs them
MAX_CHAIN = 4  # clicks in a chain that grows a tile into a fall's way
RISK = 10  # the price of landing on a look not yet stood on
MAX_STATES = 40_000
MAX_GOAL_STATES = 1_000_000  # a search for the gem made longer, once for each map
MAX_FAILS = 8
MAX_LEVEL_ACTIONS = 400
RARE = 40  # pixels a colour of the avatar's may cover in a frame


# --- reading the screen ------------------------------------------------------------


def find_phase(grid: np.ndarray, p: int) -> tuple[int, int]:
    """The lattice offset (x, y) of the cells' insides: the one with the fewest kinds of cell."""
    H, W = grid.shape
    n = p - 1
    best = None
    for oy in range(p):
        for ox in range(p):
            seen = set()
            for y in range(oy, H - 1 - n + 1, p):
                for x in range(ox, W - n + 1, p):
                    seen.add(grid[y:y + n, x:x + n].tobytes())
            if best is None or len(seen) < best[0]:
                best = (len(seen), ox, oy)
    return best[1], best[2]


def frame_shift(a: np.ndarray, b: np.ndarray) -> int:
    """How many pixels the picture moved up from frame a to frame b (the last row left out)."""
    H = a.shape[0] - 1
    best = None
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        if s >= 0:
            m = np.count_nonzero(a[s:H] != b[:H - s]) / max(1, (H - s) * a.shape[1])
        else:
            m = np.count_nonzero(a[:H + s] != b[-s:H]) / max(1, (H + s) * a.shape[1])
        if best is None or m < best[0] - 1e-9:
            best = (m, s)
    return best[1]


def rare_colours(grid: np.ndarray) -> dict:
    """Colours that cover only a few pixels: colour -> its bounding box (x1, y1, x2, y2)."""
    vals, counts = np.unique(grid[:-1], return_counts=True)
    out = {}
    for v, n in zip(vals.tolist(), counts.tolist()):
        if n <= RARE:
            ys, xs = np.nonzero(grid[:-1] == v)
            out[v] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return out


# --- what is known of a level ------------------------------------------------------


class Board:
    """The map of a level as far as it is known, and the physics on it."""

    def __init__(self, cells: dict, know: "Know", rows: tuple, view: dict, width: int, goal_cells: set) -> None:
        self.cells = cells  # (x, y) -> look id, for every cell seen
        self.know = know
        self.rows = rows  # the rows seen: (first, last)
        self.view = view  # gravity -> (rows above, rows below) the avatar in view
        self.width = width
        self.goal_cells = goal_cells
        self._dicts: dict = {}

    def look(self, c, changes: dict):
        v = changes.get(c)
        return v if v is not None else self.cells.get(c)

    def changes(self, frozen: frozenset) -> dict:
        """A state's changes as a dict, made once for each set of them."""
        d = self._dicts.get(frozen)
        if d is None:
            if len(self._dicts) > 20_000:
                self._dicts.clear()
            d = self._dicts[frozen] = dict(frozen)
        return d

    def changed(self, frozen: frozenset, now: dict, new: dict) -> frozenset:
        """The changes once the cells in ``new`` take their looks; a cell back as mapped is no
        change."""
        gone = {(c, now[c]) for c in new if c in now}
        come = {(c, L) for c, L in new.items() if self.cells.get(c) != L}
        return frozen.difference(gone).union(come) if gone or come else frozen

    def open(self, look) -> bool:
        return look is not None and self.know.passable(look)

    def fall(self, start, g: int, changes: dict):
        """From start, fall the way g pulls: (resting cell, what it rests against: a look,
        "goal", "unseen" where the fall goes on into rows not yet seen, or None where the map
        has no look)."""
        c = start
        for _ in range(200):
            n = (c[0], c[1] + g)
            if n in self.goal_cells:
                return n, "goal"
            L = self.look(n, changes)
            if L is None:
                if not self.rows[0] <= n[1] <= self.rows[1]:
                    return c, "unseen"
                return c, None
            if self.know.passable(L):
                c = n
                continue
            return c, L
        return c, None

    def visible(self, c, at, g: int) -> bool:
        up, down = self.view[g]
        return 0 <= c[0] < self.width and at[1] - up <= c[1] <= at[1] + down

    def unseen_in_view(self, at, g: int) -> bool:
        up, down = self.view[g]
        return at[1] - up < self.rows[0] or at[1] + down > self.rows[1]


class Know:
    """What the looks do, for the whole game."""

    def __init__(self) -> None:
        self.bg: int | None = None  # the background's colour
        self.open_guess: dict = {}  # look -> taken to be open (drawn mostly in the background's colour)
        self.open_seen: dict = {}  # look -> open as seen
        # look -> ("become", look) | ("grow", look) | ("gravity", look) | ("nothing", None);
        # a tile that grows becomes the look, and its look grows into every empty cell beside it
        self.effect: dict = {}
        self.safe: set = set()
        self.deadly: set = set()
        self.goal: int | None = None
        self.not_goal: set = set()

    def passable(self, look) -> bool:
        if look in self.open_seen:
            return self.open_seen[look]
        return self.open_guess.get(look, False)


# --- the solver --------------------------------------------------------------------


class GravitySolver:
    name = "gravity"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.know = Know()
        self.looks: dict = {}  # pixels -> look id, for the game
        self.pixels: dict = {}  # look id -> pixels
        self.avatar_colours: set | None = None  # kept for the game
        self.pitch: int | None = None
        self.screen: dict = {}  # gravity -> where the avatar's cell rests on screen
        self.g0_seen = None
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.others = False
        self.ready = False  # the lattice and the avatar are known on this level
        self.probe = None  # the first move, to find the avatar
        self.queue: list = []
        self.expect = None
        self.actions = 0
        self.fails = 0
        self.cells: dict = {}  # (x, y) -> look id as now
        self.first: dict = {}  # (x, y) -> look id as first seen (the level as it began)
        self._long_search = None  # what was known when the gem was last searched for at length
        self.rows = (0, 0)
        self.last_grid = None
        self.reset_last = False
        self.trying = None  # a look taken for the gem while no gem is known

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    def _fail(self, why: str) -> None:
        self.fails += 1
        self.stats[why] += 1
        self.queue = []

    def _look_id(self, block: np.ndarray) -> int:
        key = block.tobytes()
        i = self.looks.get(key)
        if i is None:
            i = len(self.looks)
            self.looks[key] = i
            self.pixels[i] = block.copy()
            if self.know.bg is not None:
                self.know.open_guess[i] = float((block == self.know.bg).mean()) >= 0.5
        return i

    # --- reading -----------------------------------------------------------------

    def _read(self, grid: np.ndarray, only_new: bool = False) -> None:
        """Every cell wholly in view, into the map; the avatar's cell is open. ``only_new``:
        only cells not in the map yet, none with the avatar's main colour in it (a frame in
        the middle of an action, or one the avatar is not in)."""
        p, n = self.pitch, self.pitch - 1
        H = grid.shape[0]
        first_row = -((self.off) // p)
        rows = []
        if not only_new:
            self.in_view = set()
        for cy in range(first_row - 1, first_row + H // p + 2):
            top = self.off + cy * p
            if top < 0 or top + n > H - 1:
                continue
            rows.append(cy)
            for cx in range(self.width):
                c = (cx, cy)
                if c == self.at or only_new and c in self.cells:
                    continue
                x = self.ox + cx * p
                block = grid[top:top + n, x:x + n]
                if only_new and (block == self.avatar_mark[0]).any():
                    continue
                L = self._look_id(block)
                self.cells[c] = L
                self.first.setdefault(c, L)
                if not only_new:
                    self.in_view.add(c)
        if rows:
            self.rows = (min(self.rows[0], rows[0]), max(self.rows[1], rows[-1]))

    def _avatar_spot(self, grid: np.ndarray):
        """The corner of the avatar's cell's inside on screen: where its main colour is drawn
        in as many pixels, and as wide and tall, as the first move showed (the same colour may
        be drawn elsewhere)."""
        colour, count, w, h = self.avatar_mark
        found = []
        for comp in fp.segment(np.where(grid[:-1] == colour, 1, 0), connectivity=8):
            if comp.color == 1 and comp.area == count and comp.width == w and comp.height == h:
                found.append(((comp.x1 + comp.x2) // 2, (comp.y1 + comp.y2) // 2))
        if len(found) != 1:
            return None
        x, y = found[0]  # the cell is the one its middle lies in
        p = self.pitch
        oy = self.off % p
        return self.ox + ((x - self.ox) // p) * p, oy + ((y - oy) // p) * p

    def _start(self, grid: np.ndarray) -> bool:
        """The level's lattice, avatar and map, from a frame of its start (the avatar's
        colours known): the avatar's row is row 0."""
        p = self.pitch
        self.dtype, self.h = grid.dtype, grid.shape[0]
        self.ox, oy = find_phase(grid, p)
        self.width = (grid.shape[1] - self.ox - (p - 1)) // p + 1
        self.off = oy
        spot = self._avatar_spot(grid)
        if spot is None:
            return False
        sx, sy = spot
        self.off = sy
        self.at = ((sx - self.ox) // p, 0)
        self.start_at = self.at
        if self.know.bg is None:
            uniform = Counter()
            for y in range(oy, grid.shape[0] - p, p):
                for x in range(self.ox, grid.shape[1] - p + 1, p):
                    b = grid[y:y + p - 1, x:x + p - 1]
                    if (b == b.flat[0]).all():
                        uniform[int(b.flat[0])] += 1
            if not uniform:
                return False
            self.know.bg = uniform.most_common(1)[0][0]
            for i, px in self.pixels.items():
                self.know.open_guess[i] = float((px == self.know.bg).mean()) >= 0.5
        self._read(grid)
        up = self.cells.get((self.at[0], -1))
        down = self.cells.get((self.at[0], 1))
        up = up is not None and not self.know.passable(up)
        down = down is not None and not self.know.passable(down)
        if up != down:
            self.g = -1 if up else 1  # it rests against the solid side
        elif self.g0_seen is not None and sy in self.screen.values():
            self.g = next(g for g, top in self.screen.items() if top == sy)
        else:
            self.g = 1 if sy < grid.shape[0] // 2 else -1  # the view leaves room the way it falls
        self.g0 = self.g0_seen = self.g
        self.screen[self.g] = sy  # where the avatar rests on screen, by gravity
        rest = self.cells.get((self.at[0], self.at[1] + self.g))
        if rest is not None:
            self.know.safe.add(rest)
        self.last_grid = grid.copy()
        self.ready = True
        return True

    def _top(self, g: int) -> int:
        """Where the avatar's cell's inside rests on screen when gravity is g."""
        if g in self.screen:
            return self.screen[g]
        return (self.h - 1 - (self.pitch - 1)) - self.screen[-g]  # mirrored, until seen

    def _board(self) -> Board:
        p, n = self.pitch, self.pitch - 1
        view = {}
        for g in (-1, 1):
            top = self._top(g)
            view[g] = (top // p, (self.h - 1 - n - top) // p)
        goals = set()
        if self.know.goal is not None:
            goals = {c for c, L in self.cells.items() if L == self.know.goal}
        b = Board(self.cells, self.know, self.rows, view, self.width, goals)
        b.air = self._air()
        b.growers = {L for L, e in self.know.effect.items() if e[0] == "grow"}
        return b

    # --- planning -------------------------------------------------------------------

    def _column(self, b: Board, changes: dict, start, g: int, at):
        """Ways to make a fall from start go on or stop: lists of cells to click first."""
        opts = [[]]
        pre: list = []
        c = (start[0], start[1] + g)
        for i in range(60):
            if c in b.goal_cells:
                break
            L = b.look(c, changes)
            if L is None:
                break
            eff = b.know.effect.get(L)
            if b.open(L):
                if eff and eff[0] == "become" and not b.open(eff[1]) and len(pre) < MAX_PRE and b.visible(c, at, g):
                    opts.append(pre + [c])  # turned solid: the fall stops before it
                elif b.growers and i < MAX_CHAIN and not pre:
                    chain = self._chain(b, changes, c, at, g)
                    if chain:
                        opts.append(chain)  # a tile grown into it: the fall stops before it
                c = (c[0], c[1] + g)
                continue
            if eff and eff[0] in ("become", "grow") and b.open(eff[1]) and len(pre) < MAX_PRE and b.visible(c, at, g):
                pre = pre + [c]  # cleared: the fall goes on
                opts.append(list(pre))
                c = (c[0], c[1] + g)
                continue
            break
        return opts

    def _chain(self, b: Board, changes: dict, target, at, g: int):
        """The fewest clicks that grow a tile into the empty cell ``target``: a tile that grows,
        then each empty cell on the way to it, all in view; None if no chain is short enough."""
        sup = (at[0], at[1] + g)
        prev = {target: None}
        frontier = [target]
        for _ in range(MAX_CHAIN):
            nxt = []
            for e in frontier:
                for n in ((e[0] + 1, e[1]), (e[0] - 1, e[1]), (e[0], e[1] + 1), (e[0], e[1] - 1)):
                    if n in prev or n == at or n == sup or not b.visible(n, at, g):
                        continue
                    L = b.look(n, changes)
                    if L in b.growers:
                        clicks, k = [n], e
                        while k != target:
                            clicks.append(k)
                            k = prev[k]
                        return clicks
                    if L == b.air:
                        prev[n] = e
                        nxt.append(n)
            frontier = nxt
        return None

    def _apply(self, b: Board, state: tuple, a):
        """One action on the model: (state, outcome), the outcome "goal", "still", or what the
        avatar came to rest against (a look, or None where the map runs out); None when the
        action cannot be made."""
        at, g, changes = state
        ch = b.changes(changes)
        if a[0] == "move":
            t = (at[0] + a[1], at[1])
            if t in b.goal_cells:
                return (t, g, changes), "goal"
            if not b.open(b.look(t, ch)):
                return None
            land, against = b.fall(t, g, ch)
            return (land, g, changes), against
        c = a[1]
        if not b.visible(c, at, g):
            return None
        eff = b.know.effect.get(b.look(c, ch))
        if eff is None or eff[0] == "nothing":
            return None
        kind, to = eff
        if kind in ("become", "grow"):
            new = {c: to}
            if kind == "grow":
                L0 = b.look(c, ch)
                for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
                    if n != at and 0 <= n[0] < b.width and b.look(n, ch) == b.air:
                        new[n] = L0
            frozen = b.changed(changes, ch, new)
            if c == (at[0], at[1] + g) and b.open(to):
                land, against = b.fall(c, g, b.changes(frozen))
                return (land, g, frozen), against
            return (at, g, frozen), "still"
        # gravity turns: the fall is made with the tile still there, then it changes
        g2 = -g
        sup = (at[0], at[1] + g2)
        land, against = at, b.look(sup, ch)
        if sup in b.goal_cells:
            against = "goal"
        elif b.open(against):
            land, against = b.fall(at, g2, ch)
        return (land, g2, b.changed(changes, ch, {c: to})), against

    def _macros(self, b: Board, state: tuple, turners: list, sealed: frozenset = frozenset()):
        at, g, changes = state
        ch = b.changes(changes)
        out = []
        for d in (-1, 1):
            t = (at[0] + d, at[1])
            L = b.look(t, ch)
            heads = [[]]
            if t not in b.goal_cells and not b.open(L):
                eff = b.know.effect.get(L)
                if not (eff and eff[0] in ("become", "grow") and b.open(eff[1]) and b.visible(t, at, g)):
                    continue
                heads = [[t]]
            for h in heads:
                ch2 = dict(ch)
                for c in h:
                    ch2[c] = b.know.effect[b.look(c, ch)][1]
                for col in self._column(b, ch2, t, g, at):
                    out.append([("click", c) for c in h + col] + [("move", d)])
        sup = (at[0], at[1] + g)
        eff = b.know.effect.get(b.look(sup, ch))
        if eff and eff[0] in ("become", "grow") and b.open(eff[1]) and b.visible(sup, at, g):
            for col in self._column(b, ch, sup, g, at):
                out.append([("click", c) for c in col] + [("click", sup)])
        took_sealed = False
        for c in turners:
            eff = b.know.effect.get(b.look(c, ch))
            if eff and eff[0] == "gravity" and b.visible(c, at, g):
                if c in sealed:
                    if took_sealed:
                        continue  # one sealed turner does what any other would
                    took_sealed = True
                for col in self._column(b, ch, at, -g, at):
                    out.append([("click", cc) for cc in col] + [("click", c)])
        return out

    def _sealed(self, b: Board, turners: list) -> frozenset:
        """Turners shut in: every cell next to them, or to the turners they touch, is a wall, a
        deadly look, a look whose click does nothing or the board's edge. Which of them have
        been clicked away never matters to a way, so the search's states leave them out."""
        tset, seen, sealed = set(turners), set(), set()
        wall = self._wall()
        for c0 in turners:
            if c0 in seen:
                continue
            group, stack, shut = [], [c0], True
            seen.add(c0)
            while stack:
                c = stack.pop()
                group.append(c)
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (c[0] + d[0], c[1] + d[1])
                    if n in tset:
                        if n not in seen:
                            seen.add(n)
                            stack.append(n)
                        continue
                    if not 0 <= n[0] < b.width:
                        continue
                    L = b.cells.get(n)
                    if L is None and not b.rows[0] <= n[1] <= b.rows[1]:
                        continue  # rows not seen yet: no way the search knows of goes through them
                    if L is not None and (L == wall or L in self.know.deadly
                                          or self.know.effect.get(L, ("",))[0] == "nothing"):
                        continue
                    shut = False
            if shut:
                sealed.update(group)
        return frozenset(sealed)

    def _search(self, explore: int, cap: int = MAX_STATES):
        """The cheapest steps to the gem, or exploring, to a fall on into rows not yet seen
        (explore 1) or a view of some (explore 2): a list of (action, state before, state
        after, outcome)."""
        b = self._board()
        start = (self.at, self.g, frozenset())
        if explore == 2 and b.unseen_in_view(self.at, self.g):
            return None
        turners = [c for c, L in b.cells.items() if b.know.effect.get(L, ("",))[0] == "gravity"]
        sealed = self._sealed(b, turners)
        # Tiles that turn both ways (toggles) are set just before the step that needs them,
        # so which way they stand is left out of the states too, as are sealed turners.
        eff = self.know.effect
        two_way = {L for L, e in eff.items() if e[0] == "become" and eff.get(e[1], ("", None))[1] == L}

        loose = two_way | b.growers  # what grows, grows again round the avatar

        keys: dict = {}

        def key(st):
            k = keys.get(st[2])
            if k is None:
                k = keys[st[2]] = frozenset(x for x in st[2] if x[0] not in sealed and x[1] not in loose
                                            and b.cells.get(x[0]) not in loose)
            return (st[0], st[1], k)

        best = {key(start): 0}
        prev = {key(start): None}
        heap = [(0, 0, start, None)]
        n = 0
        while heap:
            cost, _, state, end = heapq.heappop(heap)
            if end is not None:  # the cheapest way there
                k, path = end
                while prev[key(k)] is not None:
                    k, ss = prev[key(k)]
                    path = ss + path
                return path
            if cost > best.get(key(state), 1 << 30):
                continue
            for mac in self._macros(b, state, turners, sealed):
                cur, extra, ok, won = state, 0, True, False
                steps = []
                for a in mac:
                    r = self._apply(b, cur, a)
                    if r is None:
                        ok = False
                        break
                    nxt, out = r
                    steps.append((a, cur, nxt, out))
                    cur = nxt
                    if out == "goal":
                        won = True
                        break
                    if out in self.know.deadly:
                        ok = False
                        break
                    if out == "unseen":
                        break
                    if out not in (None, "still") and out not in self.know.safe:
                        extra += RISK
                if not ok:
                    continue
                total = cost + len(steps) + extra
                if explore == 1 and steps[-1][3] == "unseen" or explore == 2 and b.unseen_in_view(cur[0], cur[1]):
                    won = steps[-1][3] == "unseen" or self._way_on(b, cur, turners)
                n += 1
                if won:
                    heapq.heappush(heap, (total, n, cur, (state, list(steps))))
                elif total < best.get(key(cur), 1 << 30):
                    best[key(cur)] = total
                    prev[key(cur)] = (state, steps)
                    heapq.heappush(heap, (total, n, cur, None))
            if n > cap:
                return None
        return None

    def _way_on(self, b: Board, state: tuple, turners: list) -> bool:
        """Whether the avatar can go on from a state, onto something it has stood on or into
        rows not seen yet: not a dead end, nor one whose only way on is an untried landing."""
        for mac in self._macros(b, state, turners):
            cur, ok, out = state, True, None
            for a in mac:
                r = self._apply(b, cur, a)
                if r is None or r[1] in self.know.deadly:
                    ok = False
                    break
                cur, out = r
            if ok and cur[:2] != state[:2] and (out in ("unseen", "goal", None) or out in self.know.safe):
                return True
        return False

    def _air(self) -> int:
        """The look of an empty cell: all background."""
        n = self.pitch - 1
        return self._look_id(np.full((n, n), self.know.bg, dtype=self.dtype))

    def _wall(self):
        """The commonest look of the level but the background's: the walls."""
        counts = Counter(L for L in self.first.values() if not (self.pixels[L] == self.know.bg).all())
        return counts.most_common(1)[0][0] if counts else None

    def _learn_click(self):
        """A look in view whose click has not been seen: one of it, away from the avatar."""
        b = self._board()
        sup = (self.at[0], self.at[1] + self.g)
        wall = self._wall()
        cands = []
        for c, L in self.cells.items():
            if L in self.know.effect or L == wall or L == self.know.goal or L in self.know.deadly or c == sup:
                continue
            if (self.pixels[L] == self.know.bg).all() or not b.visible(c, self.at, self.g):
                continue
            cands.append((c[0] == self.at[0], abs(c[0] - self.at[0]) + abs(c[1] - self.at[1]), c))
        if not cands:
            return None
        return min(cands)[2]

    def _plan(self):
        self.trying = None
        gem = self.know.goal is not None and any(L == self.know.goal for L in self.cells.values())
        if self.know.goal is not None:
            if gem:
                path = self._search(0)
                if path:
                    return path
        else:
            counts = Counter(self.cells.values())
            wall = self._wall()
            for L, k in sorted(counts.items()):
                if k != 1 or L == wall or L in self.know.not_goal or L in self.know.effect or self.know.passable(L):
                    continue
                self.know.goal = L  # tried: a look seen only once
                path = self._search(0)
                self.know.goal = None
                if path:
                    self.trying = L
                    self.stats["goal_tries"] += 1
                    return path
        for explore in (2, 1):
            path = self._search(explore)
            if path:
                self.stats["explores"] += 1
                return path
        c = self._learn_click()
        if c is not None:
            self.stats["learn_clicks"] += 1
            state = (self.at, self.g, frozenset())
            return [(("click", c), state, None, None)]
        return None

    def _plan_long(self):
        """Stuck at the level's start: the same searches made longer, once for what is known."""
        known = (self.at, self.g, len(self.cells), frozenset(self.know.effect), len(self.know.deadly))
        if self._long_search == known:
            return None
        self._long_search = known
        gem = self.know.goal is not None and any(L == self.know.goal for L in self.cells.values())
        for explore in ((0, 2, 1) if gem else (2, 1)):
            self.stats["long_searches"] += 1
            path = self._search(explore, MAX_GOAL_STATES)
            if path:
                self.stats["explores"] += explore > 0
                return path
        return None

    # --- acting ----------------------------------------------------------------------

    def _to_action(self, a, state) -> Action:
        if a[0] == "move":
            return (RIGHT if a[1] > 0 else LEFT), None
        c = a[1]
        at, g, _ = state
        p = self.pitch
        x = self.ox + c[0] * p + (p - 1) // 2
        y = self._top(g) + (c[1] - at[1]) * p + (p - 1) // 2
        return CLICK, {"x": int(x), "y": int(y)}

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        avail = set(obs.available_actions)
        if not {3, 4, 6} <= avail or 1 in avail or 2 in avail:
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if not self.ready:
            if self.read or self.others:
                return None
            self.read = True
            if self.avatar_colours is None:
                # the first move shows which of the rare shapes is the avatar
                rare = rare_colours(grid)
                small = {c: b for c, b in rare.items() if b[2] - b[0] <= 8 and b[3] - b[1] <= 8}
                if not small:
                    self.gave_up = True
                    return None
                self.probe = (small, grid.copy())
                self.expect = ("probe",)
                self.actions += 1
                self.in_control = True
                self.stats["takeovers"] += 1
                return RIGHT, None
            if not self._start(grid):
                self.gave_up = True
                return None
            self.stats["scenes"] += 1
        if not self.queue:
            plan = self._plan()
            if not plan and self.at != self.start_at and not self.reset_last:
                self.stats["restarts"] += 1  # stuck: the level again, with the map kept
                self.reset_last = True
                self.expect = ("reset",)
                self.actions += 1
                return GameAction.RESET, None
            if not plan:
                plan = self._plan_long()
            if not plan:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            self.queue = list(plan)
            self.stats["plans"] += 1
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        a, before, after, out = self.queue.pop(0)
        if a[0] == "move":
            self.touched = self.cells.get((self.at[0] + a[1], self.at[1]))
        else:
            self.touched = self.cells.get(a[1])
        self.seen_before = dict(self.cells)
        self.expect = ("step", a, before, after, out, self.trying)
        self.actions += 1
        self.stats["actions"] += 1
        return self._to_action(a, before)

    # --- observing --------------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty():
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
                exp = self.expect
                if self.know.goal is None and exp is not None and exp[0] == "step" and exp[5] is not None:
                    self.know.goal = exp[5]
                    self.stats["learned_goal"] += 1
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        exp, self.expect = self.expect, None
        frames = [np.asarray(f) for f in obs.frame]
        if obs.state != GameState.NOT_FINISHED:
            if exp is not None and exp[0] == "step":
                # the go ended: what the avatar came to rest on killed it
                out = exp[4]
                if frames and self.ready:
                    if self._track(frames, alive=False):
                        out = self.cells.get((self.at[0], self.at[1] + self.g), out)
                    else:  # out of view: the fall on the map as it now stands
                        r = self._apply(self._board(), exp[2], exp[1])
                        out = r[1] if r is not None else out
                if isinstance(out, int) and out not in self.know.safe:
                    self.know.deadly.add(out)
                    self.stats["learned_deadly"] += 1
            self.queue = []
            return
        if action is GameAction.RESET:
            self.queue = []
            self.reset_last = True
            if self.pitch is not None and self.first:
                keep, rows = dict(self.first), self.rows
                if self._start(frames[-1]):
                    for c, L in keep.items():
                        self.first.setdefault(c, L)
                    self.cells = dict(self.first)
                    self.rows = (min(rows[0], self.rows[0]), max(rows[1], self.rows[1]))
                    self.read = True
            return
        if action is None:
            return
        self.reset_last = False
        if exp is None:
            if self.ready:
                self.others = True
                self.ready = False
            return
        if exp[0] == "probe":
            self._observe_probe(frames)
            return
        _, a, before, after, out, trying = exp
        if self._track(frames):
            self._check(a, before, after, out, trying)

    def _observe_probe(self, frames: list) -> None:
        small, before = self.probe
        self.probe = None
        after = {c: b for c, b in rare_colours(frames[-1]).items() if b[2] - b[0] <= 8 and b[3] - b[1] <= 8}
        moved: dict = {}
        for c, b0 in small.items():
            b1 = after.get(c)
            if b1 is not None and b1[0] != b0[0] and b1[1] == b0[1]:
                moved.setdefault(b1[0] - b0[0], set()).add(c)
        if len(moved) != 1:
            self._give_up()
            return
        dx, colours = next(iter(moved.items()))
        self.avatar_colours = colours
        self.pitch = abs(dx)
        grid = frames[-1]
        main = max(colours, key=lambda c: int((grid[:-1] == c).sum()))
        box = after[main]
        # the main colour's patch: as many pixels, as wide and as tall
        self.avatar_mark = (main, int((grid[:-1] == main).sum()), box[2] - box[0] + 1, box[3] - box[1] + 1)
        if not self._start(before):
            self._give_up()
            return
        self.stats["probe"] += 1
        self.stats["scenes"] += 1
        self._track(frames)  # the probe's move, as any other

    def _track(self, frames: list, alive: bool = True) -> bool:
        """Where the view moved and the avatar went; the map from what is now in view. False
        when the avatar is not found."""
        prev = self.last_grid if self.last_grid is not None else frames[0]
        s = read_at = 0
        off0 = self.off
        for i, f in enumerate(frames):
            s += frame_shift(prev, f)
            prev = f
            if s != read_at and i < len(frames) - 1:
                self.off, read_at = off0 - s, s  # rows that came into view on the way
                self._read(f, only_new=True)
        self.last_grid = frames[-1].copy()  # the game draws its next frames over this one
        self.off = off0 - s
        grid = frames[-1]
        spot = self._avatar_spot(grid)
        if spot is None:
            self._read(grid, only_new=True)  # what came into view, all the same
            self._fail("avatar_lost")
            return False
        sx, sy = spot
        if sy != self.screen.get(self.g):
            self.g = -self.g  # the avatar rests elsewhere on screen: gravity has turned
            self.screen.setdefault(self.g, sy)
        p = self.pitch
        self.at = ((sx - self.ox) // p, (sy - self.off) // p)
        self._read(grid)
        rest = self.cells.get((self.at[0], self.at[1] + self.g))
        if rest is not None and alive:
            self.know.safe.add(rest)
        return True

    def _check(self, a, before, after, out, trying) -> None:
        """The model against what happened: learn what differs, plan again if it did."""
        if a[0] == "click":
            c, L0 = a[1], self.touched
            seen = c in self.in_view
            L1 = self.cells.get(c) if seen else None
            if L0 is not None and L0 not in self.know.effect:
                # a tile that went out of view as the avatar fell is taken to have vanished
                to = L1 if seen else self._air()
                if self.g != before[1]:
                    self.know.effect[L0] = ("gravity", to)
                    self.stats["learned_gravity"] += 1
                elif seen and L1 != L0 or not seen and self.at != before[0]:
                    air = self._air()
                    grew = [n for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1))
                            if self.seen_before.get(n) == air and self.cells.get(n) == L0]
                    self.know.effect[L0] = ("grow" if grew else "become", to)
                    self.stats["learned_grow" if grew else "learned_become"] += 1
                elif seen:
                    self.know.effect[L0] = ("nothing", None)
                    self.stats["learned_nothing"] += 1
                self.queue = []
            elif not seen and self.know.effect.get(L0, ("nothing",))[0] != "nothing":
                self.cells[c] = self.know.effect[L0][1]  # out of view: as the click makes it
        if out == "goal" and trying is not None:
            self.know.not_goal.add(trying)  # reached, and the level went on
            self.stats["not_goal"] += 1
            self.queue = []
        if after is not None and (self.at, self.g) != (after[0], after[1]):
            self._learn_motion(a, before)
            self._fail("mismatch")

    def _learn_motion(self, a, before) -> None:
        """A move or fall that went otherwise than the model said: what is open or solid."""
        at0 = before[0]
        old = self.seen_before
        if a[0] == "move":
            if self.at == at0:
                if self.touched is not None:
                    self.know.open_seen[self.touched] = False
                    self.stats["learned_solid"] += 1
                return
            if self.touched is not None and self.at[0] == at0[0] + a[1]:
                self.know.open_seen[self.touched] = True
        x = self.at[0]
        lo, hi = sorted((at0[1], self.at[1]))
        clicked = a[1] if a[0] == "click" else None  # its look before the click says nothing
        for y in range(lo, hi + 1):
            L = old.get((x, y))
            if L is not None and (x, y) != at0 and (x, y) != clicked and not self.know.passable(L):
                self.know.open_seen[L] = True
                self.stats["learned_open"] += 1
        stop = old.get((x, self.at[1] + self.g))
        if stop is not None and self.know.passable(stop):
            self.know.open_seen[stop] = False
            self.stats["learned_solid"] += 1
