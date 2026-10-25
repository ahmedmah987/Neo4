"""The spell family: walk the avatar to the exit, casting the spells drawn on a click pad.

Some puzzles show a maze, an avatar (a square in two colours, split across or down its
middle), an exit (a box drawn in the avatar's two colours) and, apart, a pad of 3x3 small
cells that clicks toggle. When the cells toggled on make one of the level's spells, it is
cast and the pad clears: the avatar shrinks or grows (and passes a narrow gap, or not), or
jumps to a marker, or something leaves it along the way it last moved and clears barriers.
On the first levels the pad shows the spell to draw, its cells in a colour of their own.
A move takes the avatar its own size along (a large one that is blocked tries half that);
it ends the level when it reaches the exit (a large one whose full step touches the exit is
taken to try the half step, until a level is seen to end on the full step).

From the frames alone:

* the avatar is the one square made of two equal rectangles of two colours; its colours
  are kept for the game. The floor is the colour round it, the wall the commonest other
  colour (a colour on the screen's edges is not taken for the floor), the exit the box of the avatar's colours elsewhere;
* the pad is nine equal one-colour squares on a 3x3 lattice; the spell shown is the cells
  of the colour fewer of them have, and a cell of a third colour is toggled on;
* what a spell does is learned when it is first cast, from the avatar and the screen before
  and after: a new size (and the size it goes back to), a new place (a jump, kept for the
  level and the size), cells that became floor (a shot), or nothing (a shot that met
  nothing on its way: tried again aimed elsewhere).

The plan: a search over the avatar's place and size, for the fewest actions to the exit,
by moves and by the spells whose outcome is known (a cast costs a click a cell). When there
is none, the spell shown is cast to see what it does; a shot is aimed by a move (or a move
into a wall) the way that meets something on the screen that is neither wall nor floor.
"""

from __future__ import annotations

import heapq
from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
MAX_FAILS = 8
MAX_LEVEL_ACTIONS = 200
MAX_STATES = 200_000


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


def halves(grid: np.ndarray, look: tuple, s: int) -> list:
    """Every s x s square made of two halves, one of each of the two colours: [(x, y)]."""
    H, W = grid.shape
    h = s // 2
    out = set()
    for a, b in (look, look[::-1]):
        sat = {}
        for c in (a, b):
            m = np.zeros((H + 1, W + 1), dtype=np.int32)
            m[1:, 1:] = (grid == c).cumsum(0).cumsum(1)
            sat[c] = m

        def full(c, dy, dx, rows, cols):
            m = sat[c]
            n = H - s + 1, W - s + 1
            y0, x0 = dy, dx
            t = (m[y0 + rows:y0 + rows + n[0], x0 + cols:x0 + cols + n[1]] - m[y0:y0 + n[0], x0 + cols:x0 + cols + n[1]]
                 - m[y0 + rows:y0 + rows + n[0], x0:x0 + n[1]] + m[y0:y0 + n[0], x0:x0 + n[1]])
            return t == rows * cols

        if H < s or W < s:
            continue
        across = full(a, 0, 0, h, s) & full(b, h, 0, h, s)
        down = full(a, 0, 0, s, h) & full(b, 0, h, s, h)
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
        self.jumps: dict = {}  # (pattern, size) -> where a jump lands (this level)
        self.aimed: set = set()  # (pattern, what the shot's way meets) tried without effect
        self.stops: set = set()  # (x, y, size) a move was seen not to reach

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
        if sc.exit is None or sc.floor is None or not sc.hint:
            return False
        self.scene = sc
        self.found_look = av[3]
        self.avatar = av[:3]
        self.grid = grid
        return True

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

    # --- the model -------------------------------------------------------------------

    def _free(self, x: int, y: int, s: int, passable: np.ndarray) -> bool:
        H, W = passable.shape
        if x < 0 or y < 0 or x + s > W or y + s > H:
            return False
        return bool(passable[y:y + s, x:x + s].all())

    def _at_exit(self, x: int, y: int, s: int) -> bool:
        x1, y1, x2, y2 = self.scene.exit
        return x < x2 + 1 and x + s > x1 and y < y2 + 1 and y + s > y1

    def _passable(self) -> np.ndarray:
        sc = self.scene
        m = self.grid == sc.floor
        x, y, s = self.avatar
        m[y:y + s, x:x + s] = True
        x1, y1, x2, y2 = sc.pad_box
        m[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = False
        return m

    def _step(self, st, d, passable):
        """The avatar after a key: (x, y, size), or "exit" when it reaches the exit."""
        x, y, s = st
        tries = [s]
        if s >= 4 and self.half is not False:
            tries.append(s // 2)
        for i, n in enumerate(tries):
            nx, ny = x + d[0] * n, y + d[1] * n
            if self._at_exit(nx, ny, s):
                if i + 1 < len(tries) and not self.exit_full:
                    continue  # touching the exit blocks the full step: the half step decides
                return "exit"
            if (nx, ny, s) not in self.stops and self._free(nx, ny, s, passable):
                return (nx, ny, s)
        return st

    def _cast(self, pattern, st, passable):
        """The avatar after the spell, or None when that is not known (or it changes the map)."""
        kind = self.spells.get(pattern)
        x, y, s = st
        if kind == "resize" and s in self.sizes:
            n = self.sizes[s]
            if n < s:
                return (x, y, n)
            g = n - s
            for dx in range(-g, 1):
                for dy in range(-g, 1):
                    if self._free(x + dx, y + dy, n, passable):
                        return (x + dx, y + dy, n)
            return None
        if kind == "jump" and (pattern, s) in self.jumps:
            return self.jumps[(pattern, s)] + (s,)
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
        passable = self._passable()
        start = self.avatar
        selected = sc.selected(self.grid)
        casts = [p for p in (sc.hint,) if p]
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
            for k, d in KEYS.items():
                r = self._step(st, d, passable)
                if r == "exit":
                    if win is None or g + 1 < win[0]:
                        win = (g + 1, st, ("key", k))
                    continue
                if g + 1 < cost.get(r, 1 << 30):
                    cost[r] = g + 1
                    prev[r] = (st, ("key", k))
                    tick += 1
                    heapq.heappush(heap, (g + 1, tick, r))
            for p in casts:
                r = self._cast(p, st, passable)
                if r is None:
                    continue
                c = g + len(set(p) ^ (selected if st == start else frozenset()))
                if c < cost.get(r, 1 << 30):
                    cost[r] = c
                    prev[r] = (st, ("cast", p))
                    tick += 1
                    heapq.heappush(heap, (c, tick, r))
            if len(cost) > MAX_STATES:
                break
        if win is not None:
            _, st, last = win
            return self._unroll(prev, st, [(last, "exit")], selected)
        return self._explore(prev, cost, passable, casts, selected)

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
                out.append(((how[1], None), ("at", after)))
            else:
                out.extend(self._clicks(how[1], sel, after))
                sel = frozenset()
        return out

    def _explore(self, prev, cost, passable, casts, selected):
        """Cast a spell to see what it does: at once if it was never cast, else aimed."""
        for p in casts:
            if p not in self.spells:
                self.stats["explore_cast"] += 1
                return self._clicks(p, selected, None)
        best = None
        for p in casts:
            if self.spells.get(p) not in ("dud", "shot"):
                continue
            for st, g in cost.items():
                if self.facing is not None and st == self.avatar:
                    ray = self._ray(st, self.facing, passable)
                    if self._meets(ray) and (p, ray) not in self.aimed and (best is None or g < best[0]):
                        best = (g, st, None, p, ray)
                for k, d in KEYS.items():
                    r = self._step(st, d, passable)
                    if r == "exit":
                        continue
                    ray = self._ray(r, d, passable)
                    if self._meets(ray) and (p, ray) not in self.aimed and (best is None or g + 1 < best[0]):
                        best = (g + 1, st, (k, r), p, ray)
        if best is None:
            return None
        _, st, key, p, ray = best
        self.stats["explore_aim"] += 1
        plan = self._unroll(prev, st, [], selected) if prev.get(st) is not None else []
        sel = selected
        if any(pred[0] == "cast" for _, pred in plan):
            sel = frozenset()
        if key is not None:
            plan.append(((key[0], None), ("at", key[1])))
        plan.extend(self._clicks(p, sel, None))
        self.aim = (p, ray)
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
                self._fail("no_cast")
            else:
                self._learn_cast(pattern, before, now, before_grid, grid)
                if final is None or now != final:
                    self.queue = []
        self.avatar = now
        self.grid = grid

    def _learn_cast(self, pattern, before, now, g0: np.ndarray, g1: np.ndarray) -> None:
        sc = self.scene
        x, y, s = before
        if now[2] != s:
            self.spells[pattern] = "resize"
            self.sizes[s] = now[2]
            self.sizes.setdefault(now[2], s)
            self.stats["learned_resize"] += 1
            return
        if now[:2] != (x, y):
            self.spells[pattern] = "jump"
            self.jumps[(pattern, s)] = now[:2]
            self.stats["learned_jump"] += 1
            return
        gone = (g1 == sc.floor) & (g0 != sc.floor)
        x1, y1, x2, y2 = sc.pad_box
        gone[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = False
        gone[y:y + s, x:x + s] = False
        if gone.any():
            self.spells[pattern] = "shot"
            self.stats["learned_shot"] += 1
            return
        if self.spells.get(pattern) in ("resize", "jump"):
            self.stats["cast_blocked"] += 1  # a grow with no room, say: what it does is known
            return
        if self.spells.get(pattern) != "shot":
            self.spells[pattern] = "dud"
        aim = getattr(self, "aim", None)
        if aim is not None and aim[0] == pattern:
            self.aimed.add(aim)
        elif self.facing is not None:
            self.aimed.add((pattern, self._ray(before, self.facing, self._passable())))
        self.stats["cast_no_effect"] += 1

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
        self.grid = grid
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
