"""The meet family: bodies moved at once by the same keys must be brought together.

Some puzzles show two (or more) identical bodies on a grid of cells; every key moves every
body a cell, each in its own way (one of them mirrored left to right, say), a wall stops
one body without stopping the others, and the level is done when the bodies meet (land
on one cell, or cross).

From the frames alone:

* the bodies: a colour drawn only as two to four equal solid squares; their size is the
  cell size and their corners fix the grid;
* how each body takes the keys: a probe on a level's first frame (a horizontal key), then
  every move checked against the prediction (a body that went the other way is mirrored
  on that axis);
* the floor is what the cells round the bodies are made of, walls are cells made only of
  the colours that fill the screen's margins or much of it; every other cell is unknown
  until tried. A body sent into it walks on, stays (it blocks), or everything goes back to
  where the level started: a trap, and so is every cell drawn like it;
* switches: a cell that, while a body stands on it, clears the cells of its colour
  elsewhere (gates), which come back when it leaves;
* movable blocks: an object that a click picks out (it is drawn anew) and that the keys
  then move instead of the bodies; a click elsewhere lets it go.

The plan: a search over the bodies' joint positions for the fewest keys that bring them
all together (gates open while a body stands on a switch of theirs, unknown cells never
entered). If there is none, blocks in the way are moved first, each to the nearest cell
out of the bodies' way. If there is still none, the cheapest untried thing is tried: a
cell next to a body, or a click on an object of a kind not clicked yet.
"""

from __future__ import annotations

import heapq
import itertools
from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
CLICK = GameAction.ACTION6
MAX_BODIES = 4
MAX_NODES = 200_000
MAX_JOINT = 1_500_000
MAX_FAILS = 8
MAX_LEVEL_ACTIONS = 600
MAX_TRIAL_ACTIONS = 120  # before the family is confirmed in this game
BLOCK_COST = 8  # a cell held by a movable block, priced before the block is moved
WALL_SHARE = 0.08


def add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def sig(p: np.ndarray) -> tuple:
    """What a cell looks like, whatever the cell size: one colour, a centre framed by
    another, or the colours it mixes."""
    cols = np.unique(p)
    if len(cols) == 1:
        return ("solid", int(cols[0]))
    ring = np.concatenate([p[0], p[-1], p[1:-1, 0], p[1:-1, -1]])
    centre = p[1:-1, 1:-1]
    if centre.size and len(np.unique(ring)) == 1 and len(np.unique(centre)) == 1:
        return ("framed", int(ring[0]), int(centre.flat[0]))
    return ("mixed", tuple(int(c) for c in cols))


def body_readings(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(colour, size, x origin, y origin) of every colour drawn only as two to four equal
    solid squares on one grid, largest squares first."""
    out = []
    for col in np.unique(grid):
        mask = grid == col
        n = int(mask.sum())
        if n > MAX_BODIES * 64:
            continue
        ys, xs = np.nonzero(mask)
        if len(xs) < 8:
            continue
        # equal squares, apart from each other
        seen = np.zeros_like(mask)
        squares = []
        ok = True
        for y, x in zip(ys, xs):
            if seen[y, x]:
                continue
            w = 1
            while x + w < grid.shape[1] and mask[y, x + w]:
                w += 1
            h = 1
            while y + h < grid.shape[0] and mask[y + h, x]:
                h += 1
            if w != h or not 2 <= w <= 8 or not mask[y:y + h, x:x + w].all():
                ok = False
                break
            seen[y:y + h, x:x + w] = True
            squares.append((x, y, w))
            if len(squares) > MAX_BODIES:
                ok = False
                break
        if not ok or len(squares) < 2 or seen.sum() != n:
            continue
        s = squares[0][2]
        if any(w != s for _, _, w in squares) or len({(x % s, y % s) for x, y, _ in squares}) != 1:
            continue
        # no two touching (one blob cut in squares is not two bodies)
        if any(abs(a[0] - b[0]) <= s and abs(a[1] - b[1]) <= s for a, b in itertools.combinations(squares, 2)):
            continue
        out.append((int(col), s, squares[0][0] % s, squares[0][1] % s))
    out.sort(key=lambda r: -r[1])
    return out


class Scene:
    """A frame cut into cells of the bodies' size."""

    def __init__(self, colour: int, s: int, ox: int, oy: int) -> None:
        self.colour, self.s, self.ox, self.oy = colour, s, ox, oy
        self.cells: set = set()
        for j in range(-1, 64 // s + 1):
            for i in range(-1, 64 // s + 1):
                x0, y0 = ox + i * s, oy + j * s
                if x0 >= 1 and y0 >= 1 and x0 + s <= 63 and y0 + s <= 63:
                    self.cells.add((i, j))

    def patch(self, grid: np.ndarray, c: tuple[int, int]) -> np.ndarray:
        x0, y0 = self.ox + c[0] * self.s, self.oy + c[1] * self.s
        return grid[y0:y0 + self.s, x0:x0 + self.s]

    def sigs(self, grid: np.ndarray) -> dict:
        return {c: sig(self.patch(grid, c)) for c in self.cells}

    def point(self, c: tuple[int, int]) -> tuple[int, int]:
        return self.ox + c[0] * self.s + self.s // 2, self.oy + c[1] * self.s + self.s // 2

    def bodies(self, grid: np.ndarray) -> list:
        return sorted(c for c in self.cells if (self.patch(grid, c) == self.colour).all())


class MeetSolver:
    name = "meet"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        # what kinds of cell do, learned for the game (by look, whatever the cell size)
        self.trap_sigs: set = set()
        self.shut_sigs: set = set()
        self.movable_sigs: set = set()
        self.clicked_sigs: set = set()  # clicked once already
        self.switch_colours: set = set()
        self.body_colour: int | None = None  # once a level is won with it
        self._new_level()

    # --- state -----------------------------------------------------------------------

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.fails = 0
        self.actions = 0
        self.scene: Scene | None = None
        self.queue: list = []
        self.expect: tuple | None = None
        self.stale = False
        self.can_click = False
        self.resets = 0

    def _setup(self, grid: np.ndarray, reading: tuple) -> bool:
        colour, s, ox, oy = reading
        sc = Scene(colour, s, ox, oy)
        self.scene = sc
        self.start = tuple(sc.bodies(grid))
        if not 2 <= len(self.start) <= MAX_BODIES:
            return False
        self.pos = self.start
        self.home = set(self.start)  # cells under the bodies at the start: floor
        self.ids = tuple(range(len(self.start)))
        self.sgn = [[None, 1] for _ in self.start]  # left-right probed, up-down assumed
        self.moved_ids: set = set()
        self.body_moves = 0
        self.base = sc.sigs(grid)  # what each cell looked like at the start
        near: Counter = Counter()
        for p in self.start:
            for d in KEYS.values():
                g = self.base.get(add(p, d))
                if g is not None and g[0] == "solid" and g[1] != colour:
                    near[g[1]] += 1
        if not near:
            return False
        counts = Counter(int(v) for v in grid[1:63].ravel())
        edge = {int(v) for v in grid[1:63, 0]} | {int(v) for v in grid[1:63, 63]}
        # the floor: what lies next to the bodies, the margins' colours last
        self.floor = max(near, key=lambda c: (c not in edge, near[c], c == getattr(self, "floor", None)))
        self.walls = {c for c, n in counts.items() if n >= WALL_SHARE * 62 * 64 or c in edge} - {self.floor, colour}
        self.open: set = set()  # cells learned passable for a body
        self.shut: set = set()  # cells learned to stop a body
        self.traps: set = set()
        self.block_shut: set = set()
        self.blocks = sorted(c for c, g in self.base.items() if g in self.movable_sigs)
        self.selected: tuple | None = None  # the cell of the block picked out
        self.sel_sig: tuple | None = None
        self.trying: tuple | None = None
        self.gates: dict = {}  # colour -> (switch cells, gate cells)
        for x in self.switch_colours:
            self._presume_gates(x)
        # bodies stand free: floor next to each, no untried solid cell touching any
        # (unless bodies of this colour have been played already)
        if colour != self.body_colour:
            for p in self.start:
                around = [add(p, d) for d in KEYS.values()]
                if not any(self._status(t) == "go" for t in around):
                    return False
                if any(self._status(t) == "unknown" and self.base[t][0] == "solid" for t in around):
                    return False
        return True

    def _presume_gates(self, x: int) -> None:
        """Switches of colour x are its lone cells, gates the others."""
        cells = {c for c, g in self.base.items() if g == ("solid", x)}
        lone = {c for c in cells if not any(add(c, d) in cells for d in KEYS.values())}
        if lone and cells - lone:
            self.gates[x] = (lone, cells - lone)

    # --- the model -------------------------------------------------------------------

    def _status(self, t: tuple[int, int]) -> str:
        """What a cell is for a body, gates and blocks aside: go, stop, trap or unknown."""
        if t not in self.scene.cells:
            return "stop"
        if t in self.traps:
            return "trap"
        if t in self.open:
            return "go"
        if t in self.shut:
            return "stop"
        g = self.base[t]
        if t in self.home or g == ("solid", self.floor) or any(t in sw for sw, _ in self.gates.values()):
            return "go"
        if g[0] == "solid":
            return "stop" if g[1] in self.walls else "unknown"
        cols = {g[1], g[2]} if g[0] == "framed" else set(g[1])
        if cols <= self.walls:
            return "stop"
        if g in self.trap_sigs:
            return "trap"
        if g in self.shut_sigs:
            return "stop"
        if g in self.movable_sigs:
            return "go"  # where a block stood; the block itself is counted apart
        return "unknown"

    def _gate_open(self, t: tuple[int, int], pos: tuple) -> bool | None:
        for sw, gs in self.gates.values():
            if t in gs:
                return any(p in sw for p in pos)
        return None

    def _body_cell(self, t: tuple[int, int], pos: tuple, blocks, through: bool = False) -> str:
        if t in blocks:
            return "block" if through else "stop"
        g = self._gate_open(t, pos)
        if g is not None:
            return "go" if g else "stop"
        return self._status(t)

    def _sign(self, i: int) -> tuple[int, int]:
        sx, sy = self.sgn[i]
        return (1 if sx is None else sx, 1 if sy is None else sy)

    def _targets(self, pos: tuple, ids: tuple, k: GameAction) -> list:
        d = KEYS[k]
        out = []
        for p, i in zip(pos, ids):
            sx, sy = self._sign(i)
            out.append((p[0] + d[0] * sx, p[1] + d[1] * sy))
        return out

    @staticmethod
    def _merge(before: tuple, after: list) -> tuple[list, list]:
        """Bodies that land on one cell or cross meet and drop out: (alive indices, pairs)."""
        alive = list(range(len(after)))
        pairs = []
        for i, j in itertools.combinations(range(len(after)), 2):
            if i in alive and j in alive and (after[i] == after[j] or (after[i] == before[j] and after[j] == before[i])):
                alive.remove(i)
                alive.remove(j)
                pairs.append((i, j))
        return alive, pairs

    def _step(self, pos: tuple, ids: tuple, k: GameAction, blocks, through: bool = False):
        """(new positions, cost) after key k, or None if a body would go somewhere untried."""
        new, cost = [], 1
        for p, t in zip(pos, self._targets(pos, ids, k)):
            st = self._body_cell(t, pos, blocks, through)
            if st == "go":
                new.append(t)
            elif st == "block":
                new.append(t)
                cost += BLOCK_COST
            elif st == "stop":
                new.append(p)
            else:
                return None
        return new, cost

    # --- searches --------------------------------------------------------------------

    def _search(self, blocks, through: bool = False):
        """The cheapest keys that bring every body together: (keys, cells the bodies pass)."""
        start = (self.pos, self.ids)
        best = {start: 0}
        prev: dict = {start: None}
        heap = [(0, 0, start)]
        tick = 0
        while heap:
            cost, _, st = heapq.heappop(heap)
            if cost > best[st]:
                continue
            pos, ids = st
            if not pos:
                keys, cells = [], set()
                while prev[st] is not None:
                    st0, k = prev[st]
                    keys.append(k)
                    cells |= set(st0[0]) | set(st[0])
                    st = st0
                return keys[::-1], cells
            if len(best) > MAX_NODES:
                return None
            for k in KEYS:
                r = self._step(pos, ids, k, blocks, through)
                if r is None:
                    continue
                new, c = r
                alive, pairs = self._merge(pos, new)
                ns = (tuple(new[i] for i in alive), tuple(ids[i] for i in alive))
                if pairs:
                    # the meeting cells count as passed
                    c += 0
                if ns == st:
                    continue
                nc = cost + c
                if nc < best.get(ns, 1 << 30):
                    best[ns] = nc
                    prev[ns] = (st, k)
                    tick += 1
                    heapq.heappush(heap, (nc, tick, ns))
        return None

    def _frontier(self, blocks):
        """The fewest keys after which one more sends a body into a cell not tried yet."""
        start = self.pos
        prev = {start: None}
        q = deque([start])
        while q:
            pos = q.popleft()
            for k in KEYS:
                sts = [self._body_cell(t, pos, blocks) for t in self._targets(pos, self.ids, k)]
                if "unknown" in sts and "trap" not in sts:
                    keys = [k]
                    st = pos
                    while prev[st] is not None:
                        st, kk = prev[st]
                        keys.append(kk)
                    return keys[::-1]
            if len(prev) > MAX_NODES:
                return None
            for k in KEYS:
                r = self._step(pos, self.ids, k, blocks)
                if r is None:
                    continue
                new, _ = r
                _, pairs = self._merge(pos, new)
                ns = tuple(new)
                if not pairs and ns not in prev:
                    prev[ns] = (pos, k)
                    q.append(ns)
        return None

    def _block_ok(self, t: tuple[int, int], others, pos: tuple) -> bool:
        if t not in self.scene.cells or t in self.block_shut or t in others or t in pos:
            return False
        gate = self._gate_open(t, pos)
        if gate is not None:
            return gate
        g = self.base[t]
        return t in self.open or t in self.home or g == ("solid", self.floor) or g in self.movable_sigs

    def _park(self, start: tuple[int, int], blocks: list, avoid: set):
        """The keys that take the block at start to the nearest cell out of the way."""
        others = set(blocks) - {start}
        prev = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c not in avoid:
                keys = []
                while prev[c] is not None:
                    c, k = prev[c]
                    keys.append(k)
                return keys[::-1]
            for k, d in KEYS.items():
                t = add(c, d)
                if t not in prev and self._block_ok(t, others, self.pos):
                    prev[t] = (c, k)
                    q.append(t)
        return None

    def _plan(self) -> list | None:
        blocks = list(self.blocks)
        found = self._search(frozenset(blocks))
        if found is not None:
            return ([("free",)] if self.selected is not None else []) + [("key", k) for k in found[0]]
        if not blocks or not self.can_click:
            return None
        steps: list = []
        sel = self.selected
        for _ in range(4):
            through = self._search(frozenset(blocks), through=True)
            if through is None:
                break
            moving = [c for c in blocks if c in through[1]]
            if not moving:
                break
            parked = True
            for c in moving:
                keys = self._park(c, blocks, through[1])
                if keys is None:
                    parked = False
                    break
                if sel != c:
                    steps.append(("select", c))
                t = c
                for k in keys:
                    steps.append(("bkey", k))
                    t = add(t, KEYS[k])
                blocks[blocks.index(c)] = t
                sel = t
            if not parked:
                break
            found = self._search(frozenset(blocks))
            if found is not None:
                return steps + [("free",)] + [("key", k) for k in found[0]]
        # blocks may be needed where they stop a body: search with them
        return self._joint()

    def _tables(self, pos: tuple):
        """The grid as arrays for the joint search: neighbours, what each cell is for a
        body and for a block (gates by number), and which gates each cell switches."""
        cells = sorted(self.scene.cells)
        index = {c: n for n, c in enumerate(cells)}
        gates = list(self.gates.items())
        dirs = list(KEYS.values())
        nbr = [[index.get(add(c, d), -1) for d in dirs] for c in cells]
        body, block, sw = [], [], []
        for c in cells:
            g = next((n for n, (_, (_, gs)) in enumerate(gates) if c in gs), None)
            sw.append(sum(1 << n for n, (_, (sws, _)) in enumerate(gates) if c in sws))
            if g is not None:
                body.append(10 + g)
                block.append(10 + g if c not in self.block_shut else 1)
                continue
            st = self._status(c)
            body.append({"go": 0, "stop": 1}.get(st, 2))
            block.append(0 if self._block_ok(c, (), ()) else 1)
        return cells, index, nbr, body, block, sw

    def _joint(self) -> list | None:
        """The fewest actions over bodies and blocks together: keys move the bodies, or
        the block picked out by a click; a click on the floor lets it go."""
        cells, index, nbr, body, block, sw = self._tables(self.pos)
        dirs = list(KEYS.values())
        keys = list(KEYS)
        # the way each body goes for each key
        dk = {}
        for i in set(self.ids):
            sx, sy = self._sign(i)
            dk[i] = [dirs.index((d[0] * sx, d[1] * sy)) for d in dirs]
        pos0 = tuple(index[p] for p in self.pos)
        blocks0 = tuple(index[c] for c in self.blocks)
        sel0 = blocks0.index(index[self.selected]) if self.selected in self.blocks else -1
        start = (pos0, self.ids, blocks0, sel0)
        prev: dict = {start: None}
        q = deque([start])
        while q:
            st = q.popleft()
            pos, ids, blocks, sel = st
            mask = 0
            for p in pos:
                mask |= sw[p]
            moves = []
            if sel < 0:
                for kn in range(4):
                    new = []
                    for p, i in zip(pos, ids):
                        t = nbr[p][dk[i][kn]]
                        if t < 0 or t in blocks:
                            new.append(p)
                            continue
                        c = body[t]
                        if c == 0:
                            new.append(t)
                        elif c == 1:
                            new.append(p)
                        elif c >= 10:
                            new.append(t if mask >> (c - 10) & 1 else p)
                        else:
                            new = None
                            break
                    if new is None:
                        continue
                    alive, _ = self._merge(pos, new)
                    moves.append((("key", keys[kn]), (tuple(new[n] for n in alive), tuple(ids[n] for n in alive), blocks, -1)))
            else:
                c0 = blocks[sel]
                for kn in range(4):
                    t = nbr[c0][kn]
                    if t < 0 or t in pos or t in blocks:
                        continue
                    c = block[t]
                    if c == 0 or c >= 10 and mask >> (c - 10) & 1:
                        nb = list(blocks)
                        nb[sel] = t
                        moves.append((("bkey", keys[kn]), (pos, ids, tuple(nb), sel)))
                moves.append((("free",), (pos, ids, blocks, -1)))
            for bn, c in enumerate(blocks):
                if bn != sel:
                    moves.append((("select", cells[c]), (pos, ids, blocks, bn)))
            for step, ns in moves:
                if ns in prev:
                    continue
                prev[ns] = (st, step)
                if not ns[0]:
                    steps = []
                    while prev[ns] is not None:
                        ns, step = prev[ns]
                        steps.append(step)
                    self.stats["joint_plans"] += 1
                    return steps[::-1]
                q.append(ns)
            if len(prev) > MAX_JOINT:
                self.stats["joint_too_big"] += 1
                return None
        return None

    # --- learning from a step --------------------------------------------------------

    def _read_blocks(self, grid: np.ndarray) -> None:
        sigs = self.scene.sigs(grid)
        self.blocks = sorted(c for c, g in sigs.items() if g in self.movable_sigs or (self.sel_sig is not None and g == self.sel_sig))

    def _learn_move(self, k: GameAction, before: tuple, seen: list) -> bool:
        """Match what the bodies did to what each could have done (stay, go its way, go the
        other way) and learn from the gap. False if nothing fits."""
        d = KEYS[k]
        axis = 0 if d[0] else 1
        expected = self._step(before, self.ids, k, self.blocks)
        options = []
        for p, i in zip(before, self.ids):
            sx, sy = self._sign(i)
            options.append([(p, "stay"), ((p[0] + d[0] * sx, p[1] + d[1] * sy), "pred"),
                            ((p[0] - d[0] * sx, p[1] - d[1] * sy), "alt")])
        best = None
        for combo in itertools.product(*options):
            after = [c for c, _ in combo]
            alive, pairs = self._merge(before, after)
            left = {after[i] for i in alive}
            met = set(seen) - left
            if not left <= set(seen) or len(met) > len(pairs):
                continue
            gap = 0
            for n, (c, why) in enumerate(combo):
                if why == "alt":
                    gap += 3
                elif expected is None or c != expected[0][n]:
                    gap += 1
            if best is None or gap < best[0]:
                best = (gap, combo, alive)
        if best is None:
            return False
        _, combo, alive = best
        for n, (c, why) in enumerate(combo):
            p, i = before[n], self.ids[n]
            if why == "alt":
                self.sgn[i][axis] = -self._sign(i)[axis]
                self.stats["mirror_learned"] += 1
            elif why == "pred" and self.sgn[i][axis] is None:
                self.sgn[i][axis] = self._sign(i)[axis]
            if why == "stay":
                sx, sy = self._sign(i)
                want = (p[0] + d[0] * sx, p[1] + d[1] * sy)
                if self._gate_open(want, before) is None and want not in self.blocks and self._status(want) != "stop":
                    self.shut.add(want)
                    g = self.base.get(want)
                    if g is not None and g[0] != "solid":
                        self.shut_sigs.add(g)
                    self.stats["learned_shut"] += 1
            elif c != p:
                self.moved_ids.add(i)
                if self._status(c) != "go" and self._gate_open(c, before) is None:
                    self.open.add(c)
                    self.stats["learned_open"] += 1
        self.pos = tuple(combo[n][0] for n in alive)
        self.ids = tuple(self.ids[n] for n in alive)
        return True

    def _learn_switch(self, before_sigs: dict, grid: np.ndarray, before: tuple) -> None:
        """Cells that changed besides the bodies: a switch under a body opens them."""
        now = self.scene.sigs(grid)
        moved = set(before) | set(self.pos)
        opened: dict = {}
        for c, g in now.items():
            b = before_sigs.get(c)
            if c in moved or c in self.blocks or b == g:
                continue
            if g == ("solid", self.floor) and b is not None and b[0] == "solid" and b[1] != self.scene.colour:
                opened.setdefault(b[1], set()).add(c)
        for x, cells in opened.items():
            on = {p for p in self.pos if self.base.get(p) == ("solid", x)}
            if not on:
                continue
            sw, gs = self.gates.get(x, (set(), set()))
            gs = gs | cells
            lone = {c for c, g in self.base.items() if g == ("solid", x) and c not in gs
                    and not any(add(c, d) in gs for d in KEYS.values())}
            self.gates[x] = (sw | on | lone, gs)
            self.open |= on
            self.shut -= gs
            if x not in self.switch_colours:
                self.switch_colours.add(x)
                self.stats["switches_learned"] += 1

    def _rematch(self, bodies: list) -> None:
        """Bodies read again: each to the nearest of the ones known (they look alike)."""
        if set(bodies) == set(self.start) and len(bodies) == len(self.start):
            self.pos, self.ids = self.start, tuple(range(len(self.start)))
            return
        if len(bodies) == len(self.pos):
            best = min(itertools.permutations(bodies),
                       key=lambda perm: sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(perm, self.pos)))
            self.pos = tuple(best)
            return
        self.pos = tuple(bodies)
        self.ids = tuple(range(len(bodies)))
        self.sgn = [[None, 1] for _ in bodies]
        self.start = self.pos

    # --- the solver interface --------------------------------------------------------

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    def _fail(self, why: str) -> None:
        self.fails += 1
        self.stats[why] += 1
        self.queue = []
        self.stale = True

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
                if self.scene is not None:
                    self.body_colour = self.scene.colour
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        if action is None:
            return
        exp, self.expect = self.expect, None
        if action is GameAction.RESET:
            self.queue = []
            if self.scene is not None:
                self.stale = True
                self.selected = None
            return
        if self.scene is None:
            return
        if exp is None:
            self.stale = True
            return
        grid = np.asarray(obs.frame[-1])
        sc = self.scene
        kind = exp[0]
        if kind in ("key", "try", "probe"):
            _, before, before_sigs = exp
            self.body_moves += 1
            seen = sc.bodies(grid)
            expected = self._step(before, self.ids, action, self.blocks)
            predicted = None
            if expected is not None:
                alive, _ = self._merge(before, expected[0])
                predicted = {expected[0][i] for i in alive}
            animated = len(obs.frame) > 1
            if (set(seen) == set(self.start) and len(seen) == len(self.start) and predicted != set(seen)
                    and (set(before) != set(self.start) or animated)):
                # everything went back to the start: the move led into a trap
                for t in self._targets(before, self.ids, action):
                    if self._status(t) in ("unknown", "trap") and self._gate_open(t, before) is None and t not in self.blocks:
                        self.traps.add(t)
                        self.trap_sigs.add(self.base[t])
                self.stats["traps_learned"] += 1
                self.resets += 1
                if self.resets > 6:
                    self._fail("too_many_resets")
                self.pos, self.ids = self.start, tuple(range(len(self.start)))
                self._read_blocks(grid)
                self.queue = []
                return
            if not self._learn_move(action, before, seen):
                self._fail("move_missed")
                return
            if kind == "probe" and set(self.pos) == set(before) and len(self.readings) > 1:
                # these did not move: the bodies are another colour's squares
                self.readings.pop(0)
                self._setup(self.first, self.readings[0])
                self.pos = tuple(self.scene.bodies(grid)) or self.pos
                self.stats["reread"] += 1
                return
            self._learn_switch(before_sigs, grid, before)
            if kind != "key":
                self.queue = []
            return
        if kind == "click":
            _, cell, g, before_sigs = exp
            now = sc.sigs(grid)
            self.clicked_sigs.add(g)
            if now[cell] != before_sigs[cell]:
                self.trying = (cell, now[cell], g)  # picked out: try moving it
                self.queue = [("move_try",)]
            else:
                self.stats["click_nothing"] += 1
                if any(now[c] != before_sigs[c] for c in now):
                    self.stale = True
            return
        if kind == "move_try":
            _, cell, sel, k, g0 = exp
            t = add(cell, KEYS[k])
            if sig(sc.patch(grid, t)) == sel:
                self.movable_sigs.add(g0)
                self.sel_sig = sel
                self.stats["movable_learned"] += 1
                self.blocks = sorted([c for c, g in self.base.items() if g == g0 and c != cell] + [t])
                self.selected = t
            else:
                self.selected = cell
                self.sel_sig = sel
                self.queue = [("free",)]
                self.stale = True
            return
        if kind == "select":
            _, c = exp
            g = sig(sc.patch(grid, c))
            if self.sel_sig is None or g == self.sel_sig:
                self.selected, self.sel_sig = c, g
            else:
                self._fail("select_missed")
            return
        if kind == "bkey":
            _, c, k = exp
            t = add(c, KEYS[k])
            if t in sc.cells and sig(sc.patch(grid, t)) == self.sel_sig:
                self.blocks[self.blocks.index(c)] = t
                self.blocks.sort()
                self.selected = t
            elif sig(sc.patch(grid, c)) == self.sel_sig:
                self.block_shut.add(t)
                self.stats["block_stopped"] += 1
                self.queue = []
            else:
                self._fail("block_missed")
            return
        if kind == "free":
            self.selected = None
            if set(sc.bodies(grid)) != set(self.pos):
                self._fail("free_missed")
            self._read_blocks(grid)
            return

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        avail = set(obs.available_actions)
        if not {1, 2, 3, 4} <= avail:
            return None
        self.can_click = 6 in avail
        limit = MAX_LEVEL_ACTIONS if self.confirmed else MAX_TRIAL_ACTIONS
        if self.fails >= (MAX_FAILS if self.confirmed else MAX_FAILS // 2) or self.actions >= limit:
            self.stats["gave_up"] += 1
            if not self.confirmed:
                self.disabled = True
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if self.scene is None:
            if self.read:
                return None
            self.read = True
            readings = body_readings(grid)
            if self.body_colour is not None:
                readings.sort(key=lambda r: r[0] != self.body_colour)
            self.readings = []
            for r in readings:
                if self._setup(grid, r):
                    self.readings.append(r)
            if not self.readings:
                self.scene = None
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
            self.first = grid
            self._setup(grid, self.readings[0])
        if not self.confirmed and self.body_moves >= 6 and len(self.moved_ids) < len(self.start):
            # a "body" that never moves: not this family
            self.stats["not_bodies"] += 1
            self.disabled = True
            self._give_up()
            return None
        if self.stale:
            self.stale = False
            if self.selected is None:
                bodies = self.scene.bodies(grid)
                if not bodies:
                    self._give_up()
                    return None
                self._rematch(bodies)
            self._read_blocks(grid)
        if not self.queue:
            self.queue = self._next()
            if not self.queue:
                self.stats["no_plan"] += 1
                if not self.confirmed and not self.moved_ids:
                    self.disabled = True
                self._give_up()
                return None
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        return self._play(grid)

    def _next(self) -> list:
        """What to do next: a probe, a plan, or a try."""
        blocks = frozenset(self.blocks)
        if any(self.sgn[i][0] is None for i in self.ids) and self.selected is None:
            # left-right first: a key that tells each body's way apart
            score = 0
            k = GameAction.ACTION3
            for p, i in zip(self.pos, self.ids):
                if self.sgn[i][0] is not None:
                    continue
                a = self._body_cell((p[0] - 1, p[1]), self.pos, blocks)
                b = self._body_cell((p[0] + 1, p[1]), self.pos, blocks)
                if {a, b} & {"unknown", "trap"}:
                    score -= 10
                elif a != b or a == "go":
                    score += 1
            if score > 0:
                return [("probe", k)]
            for i in self.ids:
                if self.sgn[i][0] is None:
                    self.sgn[i][0] = 1
        plan = self._plan()
        if plan:
            return plan
        if self.selected is not None:
            return [("free",)]
        # nothing known brings them together: try something
        bump = self._frontier(blocks)
        clicks = []
        if self.can_click:
            for c in sorted(self.scene.cells):
                g = self.base[c]
                if c in self.pos or g in self.clicked_sigs or g in self.trap_sigs or g in self.movable_sigs:
                    continue
                if self._status(c) == "unknown" and sig_has_object(g, self.floor, self.walls):
                    clicks.append(c)
        if clicks and (bump is None or len(bump) > 2):
            return [("click", clicks[0])]
        if bump is not None:
            return [("key", k) for k in bump[:-1]] + [("try", bump[-1])]
        return []

    def _play(self, grid: np.ndarray) -> Action | None:
        step = self.queue.pop(0)
        kind = step[0]
        sc = self.scene
        if kind in ("key", "try", "probe"):
            if self.selected is not None:
                self.queue = []
                return self._play_free(grid)
            self.expect = (kind, self.pos, sc.sigs(grid))
            return self._do((step[1], None))
        if kind == "click":
            c = step[1]
            x, y = sc.point(c)
            self.expect = ("click", c, self.base[c], sc.sigs(grid))
            return self._do((CLICK, {"x": x, "y": y}))
        if kind == "move_try":
            cell, sel, g0 = self.trying
            for k, d in KEYS.items():
                t = add(cell, d)
                if self._status(t) == "go" and t not in self.pos and t not in self.blocks and self._gate_open(t, self.pos) is None:
                    self.expect = ("move_try", cell, sel, k, g0)
                    return self._do((k, None))
            self.selected, self.sel_sig = cell, sel
            return self._play_free(grid)
        if kind == "select":
            c = step[1]
            x, y = sc.point(c)
            self.expect = ("select", c)
            return self._do((CLICK, {"x": x, "y": y}))
        if kind == "bkey":
            if self.selected is None:
                self._fail("not_selected")
                return None
            self.expect = ("bkey", self.selected, step[1])
            return self._do((step[1], None))
        return self._play_free(grid)

    def _play_free(self, grid: np.ndarray) -> Action | None:
        """A click on a floor cell away from everything lets the selection go."""
        sc = self.scene
        spot = next((c for c in sorted(sc.cells) if self.base[c] == ("solid", self.floor)
                     and c not in self.pos and c not in self.blocks
                     and sig(sc.patch(grid, c)) == ("solid", self.floor)), None)
        if spot is None:
            self._give_up()
            return None
        x, y = sc.point(spot)
        self.expect = ("free",)
        return self._do((CLICK, {"x": x, "y": y}))

    def _do(self, a: Action) -> Action:
        self.actions += 1
        self.stats["actions"] += 1
        return a


def sig_has_object(g: tuple, floor: int | None, walls: set) -> bool:
    """A cell drawn as an object: not floor, not wall."""
    if g[0] == "solid":
        return g[1] != floor and g[1] not in walls
    cols = {g[1], g[2]} if g[0] == "framed" else set(g[1])
    return not cols <= walls | {floor}
