"""The carry family: bring every block into a goal zone, carrying it there.

Some puzzles show a player on a grid of cells, blocks, and goal zones; the keys move the
player a cell (turning it to face the way it moves), a fifth picks up the block it faces
or puts down the one it carries. A carried block stays in front of the player and moves
with it. The level is done when every block lies in a goal zone, put down. Other agents
may share the grid: helpers that fetch free blocks into the goal zones, and others that
carry blocks away, which the fifth key removes when the player faces one.

From the frames alone:

* the grid: the size most small square objects share; the floor is the commonest colour;
* goal zones are rectangles outlined in one colour round another; solid regions of that
  other colour are zones too (walkable, no goal); blocks are framed squares (a ring round
  a centre of the blocks' colour); the player is the cell drawn with one marked edge, the
  side it faces;
* agents are single solid cells that move without the player; what does not move is a
  wall, or, patterned, a barrier whose passability (for the player, for a carried block)
  is learned by trying (assumed open until a move fails);
* an agent that takes a block somewhere other than a goal zone, or from the player, is
  hostile, and is removed when it can be reached.

The plan, renewed every step (the agents move every step too): remove a hostile agent if
one can be reached; else the cheapest block to bring into a goal zone (walk to it, face
it, pick it up, carry it in, put it down), or, for one the player cannot bring in, to a
cell a helper can fetch it from; else wait.
"""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
USE = GameAction.ACTION5
MAX_FAILS = 12
MAX_LEVEL_ACTIONS = 2500


# --- reading the screen ------------------------------------------------------------


def _cell_size(grid: np.ndarray, bg: int) -> tuple[int, int, int] | None:
    sizes: Counter = Counter()
    offs: Counter = Counter()
    for c in fp.segment(grid):
        if c.color == bg or c.width != c.height or not 3 <= c.width <= 8:
            continue
        sizes[c.width] += 1
    if not sizes:
        return None
    s = sizes.most_common(1)[0][0]
    for c in fp.segment(grid):
        if c.color != bg and c.width == c.height == s:
            offs[(c.x1 % s, c.y1 % s)] += 1
    (ox, oy), _ = offs.most_common(1)[0]
    return s, ox, oy


class Grid:
    """A frame cut into cells."""

    def __init__(self, grid: np.ndarray, c: int, ox: int, oy: int, bg: int) -> None:
        self.c, self.ox, self.oy, self.bg = c, ox, oy, bg
        H, W = grid.shape
        self.rows, self.cols = (H - oy) // c, (W - ox) // c
        self.px = grid

    def patch(self, cell: tuple[int, int]) -> np.ndarray:
        x, y = cell
        c = self.c
        return self.px[self.oy + y * c:self.oy + (y + 1) * c, self.ox + x * c:self.ox + (x + 1) * c]

    def cells(self):
        for y in range(self.rows):
            for x in range(self.cols):
                yield (x, y)


def framed(p: np.ndarray) -> tuple[int, int] | None:
    """(ring colour, centre colour) of a framed square, else None."""
    ring = np.concatenate([p[0], p[-1], p[1:-1, 0], p[1:-1, -1]])
    centre = p[1:-1, 1:-1]
    r, k = int(ring[0]), int(centre.flat[0])
    if (ring == r).all() and (centre == k).all() and r != k:
        return r, k
    return None


def marked(p: np.ndarray, bg: int) -> tuple[int, tuple[int, int]] | None:
    """(body colour, facing) of a cell drawn in one colour with one edge in another."""
    for edge, d in ((p[0], (0, -1)), (p[-1], (0, 1)), (p[:, 0], (-1, 0)), (p[:, -1], (1, 0))):
        e = int(edge[0])
        if not (edge == e).all():
            continue
        rest = p.copy()
        if d == (0, -1):
            rest = p[1:]
        elif d == (0, 1):
            rest = p[:-1]
        elif d == (-1, 0):
            rest = p[:, 1:]
        else:
            rest = p[:, :-1]
        b = int(rest.flat[0])
        if (rest == b).all() and b != e and b != bg and e != bg:
            return b, d
    return None


def rectangles(grid: np.ndarray, g: Grid) -> tuple[list, list]:
    """Cell-aligned outlined rectangles (goal zones) and the solid regions of their
    inside colour (other zones): lists of cell sets."""
    goals, inside = [], set()
    for comp in fp.segment(grid):
        if comp.color == g.bg or comp.width < 2 * g.c and comp.height < 2 * g.c:
            continue
        if (comp.x1 - g.ox) % g.c or (comp.y1 - g.oy) % g.c or comp.width % g.c or comp.height % g.c:
            continue
        w, h = comp.width, comp.height
        if comp.area != 2 * (w + h) - 4:
            continue
        ys, xs = comp.pixels[:, 0], comp.pixels[:, 1]
        if not ((ys == comp.y1) | (ys == comp.y2) | (xs == comp.x1) | (xs == comp.x2)).all():
            continue
        innerc = grid[comp.y1 + 1:comp.y2, comp.x1 + 1:comp.x2]
        if innerc.size == 0:
            continue
        k = int(np.bincount(innerc.ravel()).argmax())
        if k == g.bg or k == comp.color:
            continue
        inside.add(k)
        cx1, cy1 = (comp.x1 - g.ox) // g.c, (comp.y1 - g.oy) // g.c
        goals.append({(cx1 + i, cy1 + j) for i in range(w // g.c) for j in range(h // g.c)})
    zones = []
    for comp in fp.segment(grid):
        if comp.color in inside and comp.is_rectangle and comp.width >= g.c and comp.height >= g.c:
            if (comp.x1 - g.ox) % g.c or (comp.y1 - g.oy) % g.c:
                continue
            cx1, cy1 = (comp.x1 - g.ox) // g.c, (comp.y1 - g.oy) // g.c
            cells = {(cx1 + i, cy1 + j) for i in range(comp.width // g.c) for j in range(comp.height // g.c)}
            if not any(cells <= gl for gl in goals):
                zones.append(cells)
    return goals, zones


def clean(grid: np.ndarray) -> np.ndarray:
    """The frame without status bars: a line one pixel thick along an edge, half the
    frame long or more, is replaced by what lies just inside it."""
    H, W = grid.shape
    out = grid.copy()
    for c in fp.segment(grid):
        if c.height == 1 and c.width >= W // 2 and c.y1 in (0, H - 1):
            y = c.y1
            inner = 1 if y == 0 else H - 2
            out[y, c.pixels[:, 1]] = out[inner, c.pixels[:, 1]]
        elif c.width == 1 and c.height >= H // 2 and c.x1 in (0, W - 1):
            x = c.x1
            inner = 1 if x == 0 else W - 2
            out[c.pixels[:, 0], x] = out[c.pixels[:, 0], inner]
    return out


def goal_cells(g: Grid, look: tuple[int, int]) -> tuple[set, set]:
    """Goal cells (the boxes of groups of cells holding pieces of the outline) and other
    zone cells (cells all of the goals' inside colour outside them)."""
    F, Z = look
    edge, plain = set(), set()
    for cell in g.cells():
        p = g.patch(cell)
        if p.shape != (g.c, g.c):
            continue
        cols = set(np.unique(p).tolist())
        if F in cols and cols <= {F, Z}:
            edge.add(cell)
        elif cols == {Z}:
            plain.add(cell)
    goals: set = set()
    left = set(edge)
    while left:
        start = left.pop()
        group, stack = {start}, [start]
        while stack:
            x, y = stack.pop()
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    n = (x + dx, y + dy)
                    if n in left and (abs(dx) <= 1 and abs(dy) <= 1 or (n[0] + x) // 2 in range(g.cols)):
                        if abs(dx) == 2 or abs(dy) == 2:
                            mid = ((x + n[0]) // 2, (y + n[1]) // 2)
                            if mid in edge or mid in plain:
                                continue  # a gap is only bridged over something on top
                        left.discard(n)
                        group.add(n)
                        stack.append(n)
        x1, x2 = min(c[0] for c in group), max(c[0] for c in group)
        y1, y2 = min(c[1] for c in group), max(c[1] for c in group)
        goals |= {(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)}
    return goals, plain - goals


# --- the solver ----------------------------------------------------------------------


def _add(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1])


class CarrySolver:
    name = "carry"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.keys = dict(KEYS)
        self.block_colour: int | None = None
        self.player_colour: int | None = None
        self.goal_look: tuple | None = None  # (outline colour, inside colour)
        self.agent_colours: set = set()
        self.hostile_colours: set = set()
        self.helpful_colours: set = set()  # agents seen bringing a block into a goal zone
        self.unremoved: Counter = Counter()  # tries to remove an agent of a colour that failed
        self.walk_blocked: set = set()  # cell looks the player could not enter
        self.walk_open: set = set()  # patterned looks the player did enter
        self.carry_blocked: set = set()  # cell looks a carried block could not enter
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.fails = 0
        self.actions = 0
        self.g: Grid | None = None
        self.goal: set = set()
        self.zone: set = set()
        self.static: dict = {}  # cell -> look of what never moved there
        self.under: dict = {}  # cell -> look guessed under what stood there at the start
        self.memory: dict = {}  # cell -> the last look seen there
        self.stuck: set = set()  # cells a move into failed for no reason seen
        self.carrying = False
        self.aim: tuple | None = None  # the block the player is on its way to
        self.expect: tuple | None = None
        self.prev: dict | None = None
        self.held: dict = {}  # agent cell -> block cell it carries (last frame)
        self.frames_seen = 0  # frames of this level read (a lone cell that never moves is a wall)
        self.attempt = 0  # replays of this level: each after the first varies the choices
        self.rng = random.Random(0)

    # --- reading ---------------------------------------------------------------------

    def _first(self, grid: np.ndarray) -> bool:
        grid = clean(grid)
        bg = int(np.bincount(grid.ravel()).argmax())
        cs = _cell_size(grid, bg)
        if cs is None:
            return False
        g = Grid(grid, *cs, bg)
        if self.goal_look is None:
            # an outline round another colour, whole: the goals' look, kept for the game
            outlines = rectangles(grid, g)[0]
            if not outlines:
                return False
            some = sorted(outlines[0])[0]
            p = g.patch(some)
            cols = Counter(p.ravel().tolist())
            F = int(p[0, 0])
            Z = next((k for k, _ in cols.most_common() if k != F), None)
            if Z is None:
                inner = sorted(outlines[0])[len(outlines[0]) // 2]
                Z = int(np.bincount(g.patch(inner).ravel()).argmax())
            self.goal_look = (F, Z)
        self.goal, self.zone = goal_cells(g, self.goal_look)
        if not self.goal:
            return False
        centres: Counter = Counter()
        players = []
        for cell in g.cells():
            p = g.patch(cell)
            if p.shape != (g.c, g.c) or cell in self.goal or cell in self.zone:
                continue
            f = framed(p)
            if f is not None and f != self.goal_look:
                centres[f[1]] += 1
            m = marked(p, bg)
            if m is not None and (self.player_colour is None or m[0] == self.player_colour):
                players.append((cell, m))
        for cell in self.goal:
            f = framed(g.patch(cell))
            if f is not None and f != self.goal_look:
                centres[f[1]] += 1
        if not centres or len(players) != 1:
            return False
        if self.block_colour is None or self.block_colour not in centres:
            self.block_colour = centres.most_common(1)[0][0]
        self.player_colour = players[0][1][0]
        # the floor: what the cells round the player are made of (the frame's commonest
        # colour may be what lies beyond the walls)
        around: Counter = Counter()
        pc = players[0][0]
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            n = (pc[0] + d[0], pc[1] + d[1])
            if 0 <= n[0] < g.cols and 0 <= n[1] < g.rows:
                q = g.patch(n)
                if q.shape == (g.c, g.c) and (q == q.flat[0]).all():
                    around[int(q.flat[0])] += 1
        if around:
            g.bg = around.most_common(1)[0][0]
        self.g = g
        scene = self._scene(grid)
        if scene is None:
            return False
        for cell, look in scene["solid"].items():
            self.static[cell] = look
        # under what stands somewhere at the start: what lies on both sides, if the same
        for cell in set(scene["blocks"]) | set(scene["agents"]):
            for a, b in (((-1, 0), (1, 0)), ((0, -1), (0, 1))):
                la, lb = scene["solid"].get(_add(cell, a)), scene["solid"].get(_add(cell, b))
                if la is not None and lb is not None and self._kind(la) == self._kind(lb):
                    self.under[cell] = la
                    break
        self.prev = scene
        return True

    def _scene(self, grid: np.ndarray) -> dict | None:
        g = self.g
        grid = clean(grid)
        g.px = grid
        blocks, agents, solid = set(), {}, {}
        player = facing = None
        for cell in g.cells():
            p = g.patch(cell)
            if p.shape != (g.c, g.c):
                continue
            if (p == g.bg).all():
                continue
            f = framed(p)
            if f is not None and f[1] == self.block_colour:
                blocks.add(cell)
                continue
            known = (self.agent_colours | self.hostile_colours) & set(np.unique(p).tolist())
            if known and f is not None:
                # an agent drawn framed (as when the player faces it)
                agents[cell] = next(iter(known))
                continue
            m = marked(p, g.bg)
            if m is not None and m[0] == self.player_colour and set(np.unique(p).tolist()) - self._zone_colours():
                player, facing = cell, m[1]
                continue
            if cell in self.goal or cell in self.zone:
                if not (np.isin(p, list(self._zone_colours())) | (p == g.bg)).all() and (p == p.flat[0]).all():
                    agents[cell] = int(p.flat[0])
                continue
            look = p.tobytes()
            if (p == p.flat[0]).all() and int(p.flat[0]) in self._zone_colours():
                self.zone.add(cell)  # a zone cell that something stood on at the start
                continue
            if (p == p.flat[0]).all() and (int(p.flat[0]) in self.agent_colours
                                           or self.frames_seen < 2 and self._isolated(grid, cell, p)):
                agents[cell] = int(p.flat[0])
                continue
            solid[cell] = look
        if player is None:
            return None
        for cell, look in solid.items():
            self.memory[cell] = look
        return dict(player=player, facing=facing, blocks=blocks, agents=agents, solid=solid)

    def _zone_colours(self) -> set:
        return set(self.goal_look) if self.goal_look else set()

    def _isolated(self, grid: np.ndarray, cell: tuple, p: np.ndarray) -> bool:
        """A solid cell with no 4-neighbour of its colour: maybe an agent."""
        g = self.g
        k = int(p.flat[0])
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = _add(cell, d)
            if 0 <= n[0] < g.cols and 0 <= n[1] < g.rows:
                q = g.patch(n)
                if q.shape == p.shape and (q == k).all():
                    return False
        return True

    # --- the model -------------------------------------------------------------------

    def _inside(self, cell: tuple) -> bool:
        return 0 <= cell[0] < self.g.cols and 0 <= cell[1] < self.g.rows

    def _walkable(self, sc: dict, cell: tuple, ignore: set = frozenset(), sure: bool = False) -> bool:
        if not self._inside(cell) or cell in self.stuck:
            return False
        if cell in ignore or cell in sc["blocks"] or cell in sc["agents"] or cell in sc.get("hidden", ()):
            # what lies under what stands there now, as last seen
            under = self.memory.get(cell) or self.under.get(cell)
            if under is not None and (self._kind(under) in self.walk_blocked or self._solid_look(under)):
                return False
            if under is not None and sure and self._kind(under) not in self.walk_open:
                return False
        if cell in ignore:
            return True
        if cell in sc["blocks"] or cell in sc["agents"]:
            return False
        look = sc["solid"].get(cell)
        if look is None:
            return True
        if sure and self._kind(look) not in self.walk_open:
            return False  # not known to be open: not for planning what others will do
        return False if self._kind(look) in self.walk_blocked or self._solid_look(look) else True

    def _carryable(self, sc: dict, cell: tuple, ignore: set = frozenset()) -> bool:
        if not self._inside(cell):
            return False
        if cell in ignore or cell in sc["blocks"] or cell in sc["agents"] or cell in sc.get("hidden", ()):
            under = self.memory.get(cell) or self.under.get(cell)
            if under is not None and (self._kind(under) in self.carry_blocked or self._solid_look(under)):
                return False
        if cell in ignore:
            return True
        if cell in sc["blocks"] or cell in sc["agents"]:
            return False
        look = sc["solid"].get(cell)
        if look is None:
            return True
        return False if self._kind(look) in self.carry_blocked or self._solid_look(look) else True

    def _kind(self, look: bytes) -> tuple:
        """What a look is made of: its colours and how much of each (a patterned barrier
        is the same kind wherever its pattern starts)."""
        a = np.frombuffer(look, dtype=self.g.px.dtype)
        vals, counts = np.unique(a, return_counts=True)
        return tuple(zip(vals.tolist(), counts.tolist()))

    def _solid_look(self, look: bytes) -> bool:
        """A wall: one colour throughout (a patterned cell may be a barrier, tried)."""
        a = np.frombuffer(look, dtype=self.g.px.dtype)
        return bool((a == a[0]).all())

    def _walk(self, sc: dict, start: tuple, goal_fn, ignore: set = frozenset(), sure: bool = False) -> list | None:
        """Shortest key path for the player from start to a cell goal_fn accepts."""
        seen = {start}
        q = deque([(start, [])])
        while q:
            cell, path = q.popleft()
            if goal_fn(cell):
                return path
            for key, d in self.keys.items():
                n = _add(cell, d)
                if n not in seen and self._walkable(sc, n, ignore, sure):
                    seen.add(n)
                    q.append((n, path + [(key, n)]))
        return None

    def _carry(self, sc: dict, start: tuple, off: tuple, goal_fn) -> list | None:
        """Shortest key path for the player carrying a block at offset off."""
        block = _add(start, off)
        ignore = {start, block}
        seen = {start}
        q = deque([(start, [])])
        while q:
            cell, path = q.popleft()
            if goal_fn(_add(cell, off)) and path:
                return path
            for key, d in self.keys.items():
                n = _add(cell, d)
                b = _add(n, off)
                if n in seen:
                    continue
                if self._walkable(sc, n, ignore) and self._carryable(sc, b, ignore):
                    seen.add(n)
                    q.append((n, path + [(key, n)]))
        return None

    # --- choosing --------------------------------------------------------------------

    def _decide(self, sc: dict) -> Action | None:
        p, facing = sc["player"], sc["facing"]
        front = _add(p, facing)
        hostile = {c for c, k in sc["agents"].items() if k in self.hostile_colours}
        # agents move: plan past those not next to the player (hostile ones stay in the way)
        real = sc
        sc = dict(sc, agents={a: k for a, k in sc["agents"].items()
                              if abs(a[0] - p[0]) + abs(a[1] - p[1]) <= 1 or k in self.hostile_colours})
        sc["helpers"] = real["agents"]
        if self.carrying:
            if front in self.goal:
                return self._use("drop")
            path = self._into(sc, p, facing)
            if path is None:
                path = self._best_handoff(sc, p, facing)
                if path is None:
                    self.stats["stuck_carrying"] += 1
                    return self._use("drop")
                if not path:
                    return self._use("drop")
            return self._step(path[0], "carry")
        # a hostile agent in front: remove it (the player acts before the agents move)
        if front in hostile:
            return self._use("remove")
        if hostile:
            beside = [d for d in self.keys.values() if _add(p, d) in hostile]
            if beside:
                key = next(k for k, v in self.keys.items() if v == beside[0])
                return self._step((key, p), "face")
            # diagonal: it will step beside the player (the player stands in its way): wait
            if any(abs(e[0] - p[0]) == 1 and abs(e[1] - p[1]) == 1 for e in hostile):
                if front not in real["blocks"]:
                    return self._use("lurk")
            path = self._walk(sc, p, lambda c: any(_add(c, d) in hostile for d in self.keys.values()))
            if path and len(path) <= 12:
                return self._step(path[0], "walk")
        # the cheapest block to bring in, but not one a free helper is nearer to
        claimed = self._claimed(sc)
        best = None
        for b in sc["blocks"]:
            if b in self.goal or b in self.held.values() or b in claimed:
                continue
            for d in self.keys.values():
                stand = (b[0] - d[0], b[1] - d[1])
                if stand != p and not self._walkable(sc, stand):
                    continue
                to = self._walk(sc, p, lambda c, s=stand: c == s) if stand != p else []
                if to is None:
                    continue
                into = self._into(sc, stand, d)
                cost = None
                if into is not None:
                    cost = len(to) + 2 + len(into)
                else:
                    if self._handoff(sc, b) is None:
                        out = self._best_handoff(sc, stand, d)
                        if out is not None:
                            total = self._handoff(sc, _add(stand, d) if not out else _add(out[-1][1], d))
                            cost = 1000 + len(to) + 2 + len(out) + (total or 0)
                if cost is not None and self.attempt:
                    cost += self.rng.random() * 8  # a replay tries another order
                if cost is not None and self.aim is not None and b != self.aim:
                    cost += 3  # keep to the block already chosen unless another is clearly cheaper
                if cost is not None and (best is None or cost < best[0]):
                    best = (cost, stand, d, to)
        if best is None:
            self.stats["waits"] += 1
            return self._wait(sc)
        _, stand, d, to = best
        self.aim = _add(stand, d)
        if to:
            return self._step(to[0], "walk")
        if facing == d:
            return self._use("grab")
        key = next(k for k, v in self.keys.items() if v == d)
        return self._step((key, p), "face")

    def _wait(self, sc: dict) -> Action:
        """Let a step pass without touching anything: the fifth key while facing nothing
        it acts on; else turn to face something that stops the player (a move into it
        turns without moving), else step aside."""
        p, front = sc["player"], _add(sc["player"], sc["facing"])
        # out of the way first: not on or next to a free goal cell, not next to a block
        free_goal = {c for c in self.goal if c not in sc["blocks"]}

        def clear(c: tuple) -> bool:
            near = {c} | {_add(c, d) for d in self.keys.values()}
            return not near & free_goal and not (near - {c}) & sc["blocks"]

        if free_goal and not clear(p):
            path = self._walk(sc, p, clear)
            if path:
                return self._step(path[0], "walk")
        if front not in sc["blocks"] and front not in sc["agents"]:
            return self._use("wait")
        for key, d in self.keys.items():
            n = _add(p, d)
            if n not in sc["blocks"] and n not in sc["agents"] and not self._walkable(sc, n):
                return self._step((key, p), "face")
        for key, d in self.keys.items():
            n = _add(p, d)
            if self._walkable(sc, n):
                return self._step((key, n), "walk")
        return self._use("wait")

    def _claimed(self, sc: dict) -> set:
        """The blocks free helpers are on their way to: each helper not carrying, its
        nearest free block outside the goal zones, if nearer to it than to the player."""
        free = {b for b in sc["blocks"] if b not in self.goal and b not in self.held.values()}
        if not free:
            return set()
        def dists(start: tuple) -> dict:
            out = {start: 0}
            q = deque([start])
            while q:
                c = q.popleft()
                for d in self.keys.values():
                    n = _add(c, d)
                    if n not in out and self._walkable(sc, n, {start}):
                        out[n] = out[c] + 1
                        q.append(n)
            return out
        def to_block(dist: dict, b: tuple) -> int | None:
            near = [dist[_add(b, d)] for d in self.keys.values() if _add(b, d) in dist]
            return min(near) if near else None
        mine = dists(sc["player"])
        out = set()
        for h, k in sc.get("helpers", sc["agents"]).items():
            if k not in self.agent_colours or k in self.hostile_colours or h in self.held:
                continue
            dh = dists(h)
            options = [(to_block(dh, b), b) for b in free]
            options = [(n, b) for n, b in options if n is not None]
            if options:
                n, b = min(options)
                mp = to_block(mine, b)
                if mp is None or n < mp:
                    out.add(b)
        return out

    def _depth(self, sc: dict) -> dict:
        """For each free goal cell, how far it lies from the goal's edge (free cells
        outside the goal, or those holding nothing): the deep ones first, so that no block
        bars the way to another."""
        free = {c for c in self.goal if c not in sc["blocks"]}
        out: dict = {}
        q = deque()
        for c in free:
            if any(_add(c, d) not in self.goal and self._walkable(sc, _add(c, d), {sc["player"]}) for d in self.keys.values()):
                out[c] = 0
                q.append(c)
        while q:
            c = q.popleft()
            for d in self.keys.values():
                n = _add(c, d)
                if n in free and n not in out:
                    out[n] = out[c] + 1
                    q.append(n)
        return out

    def _into(self, sc: dict, start: tuple, off: tuple) -> list | None:
        """The key path carrying a block (at start + off) into the goal: to the deepest
        free goal cell it can reach, the nearest of those."""
        block = _add(start, off)
        ignore = {start, block}
        seen = {start: []}
        q = deque([start])
        while q:
            cell = q.popleft()
            for key, d in self.keys.items():
                n = _add(cell, d)
                if n in seen:
                    continue
                if self._walkable(sc, n, ignore) and self._carryable(sc, _add(n, off), ignore):
                    seen[n] = seen[cell] + [(key, n)]
                    q.append(n)
        depth = self._depth(sc)
        best = None
        for cell, path in seen.items():
            b = _add(cell, off)
            if b in self.goal and b not in sc["blocks"] and path:
                key = (-depth.get(b, 0), len(path))
                if best is None or key < best[0]:
                    best = (key, path)
        return None if best is None else best[1]

    def _helper_reach(self, sc: dict) -> dict:
        """How far helpers are from each cell (on the board as it stands, blocks and
        agents aside, since they move; only through what is known to be open)."""
        agents = sc.get("helpers", sc["agents"])
        helpers = [c for c, k in agents.items() if k not in self.hostile_colours and k in self.agent_colours]
        key = (frozenset(helpers), frozenset(sc["blocks"]), self._known())
        if getattr(self, "_reach_key", None) != key:
            bare = dict(sc, blocks=set(), agents={}, hidden=set(sc["blocks"]) | set(sc["agents"]))
            dist: dict = {}
            q = deque((h, 0) for h in helpers)
            for h in helpers:
                dist[h] = 0
            while q:
                cell, n = q.popleft()
                for d in self.keys.values():
                    m = _add(cell, d)
                    if m not in dist and self._walkable(bare, m, sure=True):
                        dist[m] = n + 1
                        q.append((m, n + 1))
            self._reach_key, self._reach = key, dist
        return self._reach

    def _known(self) -> tuple:
        return (len(self.walk_blocked), len(self.walk_open), len(self.carry_blocked), len(self.stuck))

    def _handoff(self, sc: dict, b: tuple) -> int | None:
        """How many steps a helper would take to fetch a block put down at b and carry it
        into a goal zone (walking to a neighbour of b, then carrying), or None."""
        if b in self.goal:
            return None
        reach = self._helper_reach(sc)
        if not reach:
            return None
        bare = dict(sc, blocks=set(sc["blocks"]) - {b}, agents={}, hidden=set(sc["agents"]))
        # a helper comes from the side nearest to it, and must carry the block in from there
        sides = [((b[0] - d[0], b[1] - d[1]), d) for d in self.keys.values()]
        sides = [(reach[st], st, d) for st, d in sides if st in reach]
        if not sides:
            return None
        n, stand, d = min(sides)
        taken = self._reserved(sc, exclude=b)
        path = self._carry(bare, stand, d, lambda x: x in self.goal and x not in sc["blocks"] and x not in taken)
        if path is None:
            return None
        return n + 1 + len(path)

    def _reserved(self, sc: dict, exclude: tuple) -> set:
        """The goal cells helpers will fill with the blocks they hold or are about to
        fetch: each the nearest free goal cell its block can be carried to, in turn."""
        key = (frozenset(sc["blocks"]), frozenset(self.held.items()), exclude, self._known())
        if getattr(self, "_res_key", None) == key:
            return self._res
        reach = self._helper_reach(sc)
        bare = dict(sc, agents={}, hidden=set(sc["agents"]))
        taken: set = set()
        jobs = []
        for a, b in self.held.items():
            jobs.append((a, (b[0] - a[0], b[1] - a[1])))
        for b in sc["blocks"]:
            if b in self.goal or b == exclude or b in self.held.values():
                continue
            sides = [(reach[(b[0] - d[0], b[1] - d[1])], (b[0] - d[0], b[1] - d[1]), d)
                     for d in self.keys.values() if (b[0] - d[0], b[1] - d[1]) in reach]
            if sides and not self._walk(sc, sc["player"], lambda c, b=b: abs(c[0] - b[0]) + abs(c[1] - b[1]) == 1) is not None:
                _, stand, d = min(sides)
                jobs.append((stand, d))
        for stand, d in jobs:
            path = self._carry(bare, stand, d, lambda x: x in self.goal and x not in sc["blocks"] and x not in taken)
            if path:
                taken.add(_add(path[-1][1], d))
        self._res_key, self._res = key, taken
        return taken

    def _best_handoff(self, sc: dict, start: tuple, off: tuple) -> list | None:
        """The key path carrying a block (at start + off) to where a helper takes it in
        soonest: the player's steps and the helper's together."""
        block = _add(start, off)
        ignore = {start, block}
        seen = {start: []}
        q = deque([start])
        while q:
            cell = q.popleft()
            for key, d in self.keys.items():
                n = _add(cell, d)
                if n in seen:
                    continue
                if self._walkable(sc, n, ignore) and self._carryable(sc, _add(n, off), ignore):
                    seen[n] = seen[cell] + [(key, n)]
                    q.append(n)
        best = None
        for cell, path in seen.items():
            h = self._handoff(sc, _add(cell, off))
            if h is None:
                continue
            cost = len(path) + h + (self.rng.random() * 8 if self.attempt else 0)
            if best is None or cost < best[0]:
                best = (cost, path)
        return None if best is None else best[1]

    def _step(self, move: tuple, why: str) -> Action:
        key, _ = move
        self.expect = ("move", key, why)
        return self._do(key)

    def _use(self, why: str) -> Action:
        self.expect = ("use", USE, why)
        return self._do(USE)

    def _do(self, key: GameAction) -> Action:
        self.actions += 1
        self.stats["actions"] += 1
        return key, None

    # --- watching --------------------------------------------------------------------

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
            self.expect = None
            if action is GameAction.RESET and self.g is not None:
                self.carrying = False
                self.prev = None
                self.held = {}
                self.attempt += 1
                self.rng = random.Random(self.seed * 7919 + self.level * 101 + self.attempt)
                self.stats["replays"] += 1
            return
        if self.g is None or self.expect is None:
            return
        kind, key, why = self.expect
        self.expect = None
        grid = np.asarray(obs.frame[-1])
        self.frames_seen += 1
        sc = self._scene(grid)
        prev = self.prev
        self.prev = sc
        if sc is None or prev is None:
            return
        self._agents(prev, sc)
        if kind == "move":
            d = self.keys[key]
            want = _add(prev["player"], d)
            if sc["player"] == want:
                look = prev["solid"].get(want)
                if look is not None:
                    self.walk_open.add(self._kind(look))
                return
            if sc["player"] == prev["player"]:
                if why in ("walk", "carry"):
                    # a move that failed: what stood there stops the player, or the block
                    look = prev["solid"].get(want) or self.memory.get(want) or self.under.get(want)
                    blook = prev["solid"].get(_add(want, prev["facing"])) if self.carrying else None
                    open_ = look is None or self._kind(look) in self.walk_open
                    if look is not None and not open_ and self._kind(look) not in self.walk_blocked:
                        # the player's step first: what it would stand on
                        self.walk_blocked.add(self._kind(look))
                        self.stats["learned_blocking"] += 1
                    elif self.carrying and open_ and blook is not None and self._kind(blook) not in self.carry_blocked:
                        self.carry_blocked.add(self._kind(blook))
                        self.stats["learned_blocking"] += 1
                    elif look is None and want not in self.stuck and want not in prev["agents"]:
                        self.stuck.add(want)
                        self.stats["learned_stuck"] += 1
                        if want in self.under:
                            self.walk_blocked.add(self._kind(self.under[want]))
                    else:
                        self.fails += 1
                        self.stats["move_failed"] += 1
                return
            # it went elsewhere: learn the key
            moved = (sc["player"][0] - prev["player"][0], sc["player"][1] - prev["player"][1])
            if abs(moved[0]) + abs(moved[1]) == 1:
                self.keys[key] = moved
                self.stats["keys_learned"] += 1
            return
        if kind == "use":
            front = _add(prev["player"], prev["facing"])
            if why in ("grab", "drop"):
                self.aim = None
            if why == "grab":
                # held if it is still in front of the player (it did not go to someone else)
                self.carrying = _add(sc["player"], sc["facing"]) in sc["blocks"] and front in prev["blocks"]
            elif why == "drop":
                self.carrying = False
            elif why == "remove":
                if front not in sc["agents"]:
                    self.stats["removed"] += 1
                else:
                    k = prev["agents"].get(front)
                    self.unremoved[k] += 1
                    if self.unremoved[k] >= 3:
                        self.hostile_colours.discard(k)  # this one does not go away
                    self.fails += 1

    def _agents(self, prev: dict, sc: dict) -> None:
        """Learn which solid cells move by themselves, and which take blocks away."""
        moved_from = set(prev["agents"]) - set(sc["agents"])
        for cell in set(sc["agents"]) - set(prev["agents"]):
            k = sc["agents"][cell]
            if k in self._zone_colours():
                continue
            if any(abs(cell[0] - m[0]) + abs(cell[1] - m[1]) == 1 for m in moved_from):
                self.agent_colours.add(k)
        # solid looks that moved were agents, not walls
        for cell, look in prev["solid"].items():
            if cell not in sc["solid"] and cell in self.static:
                del self.static[cell]
        held = {}
        gone_agents = set(prev["agents"]) - set(sc["agents"])
        gone_blocks = set(prev["blocks"]) - set(sc["blocks"])
        for a in sc["agents"]:
            for d in self.keys.values():
                b = _add(a, d)
                if b not in sc["blocks"]:
                    continue
                if a in self.held and self.held[a] == b:
                    held[a] = b  # still holding it, neither moved
                    break
                # moved together: the agent and the block by the same step
                for e in self.keys.values():
                    pa, pb = (a[0] - e[0], a[1] - e[1]), (b[0] - e[0], b[1] - e[1])
                    if pa in gone_agents and (pb in gone_blocks or pb == a) and b not in prev["blocks"]:
                        held[a] = b
                        break
                if a in held:
                    break
                # just picked up: a block already there, the agent now stands still beside it
                if b in prev["blocks"] and a in prev["agents"] and b not in self.held.values():
                    continue
        # an agent that lets go of a block standing in a goal zone helps
        for a, b in self.held.items():
            if b in self.goal and b in sc["blocks"] and b not in held.values():
                k = prev["agents"].get(a)
                if k is not None and k not in self.helpful_colours:
                    self.helpful_colours.add(k)
                    self.hostile_colours.discard(k)
        # an agent that carries a block out of a goal zone, or from the player, is hostile
        for a, b in held.items():
            k = sc["agents"][a]
            was = next((pb for pa, pb in self.held.items() if abs(pa[0] - a[0]) + abs(pa[1] - a[1]) <= 1), None)
            if was is not None and was in self.goal and b not in self.goal and k not in self.helpful_colours:
                if k not in self.hostile_colours:
                    self.hostile_colours.add(k)
                    self.stats["hostile"] += 1
        if self.carrying and _add(prev["player"], prev["facing"]) in prev["blocks"]:
            front = _add(sc["player"], sc["facing"])
            if front not in sc["blocks"]:
                taker = next((a for a in sc["agents"] if a in held and held[a] == front), None)
                if taker is not None and sc["agents"][taker] not in self.helpful_colours:
                    self.hostile_colours.add(sc["agents"][taker])
                    self.stats["hostile"] += 1
                self.carrying = False
        self.held = held

    # --- acting ----------------------------------------------------------------------

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4, 5} <= set(obs.available_actions):
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self.gave_up = True
            self.in_control = False
            return None
        grid = np.asarray(obs.frame[-1])
        if self.g is None:
            if self.read:
                return None
            self.read = True
            if not self._first(grid):
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
        sc = self._scene(grid)
        if sc is None:
            self.stats["lost_sight"] += 1
            self.gave_up = True
            self.in_control = False
            return None
        self.prev = sc
        if not self.in_control:
            self.in_control = True
            self.stats["takeovers"] += 1
        return self._decide(sc)
