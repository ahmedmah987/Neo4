"""The panel family: buttons on a panel beside the board step groups of floor pieces through
their states, and the avatar walks over the floor to the goal.

Some puzzles show a board of floor pieces on a background, an avatar that walks on the floor
only, a goal square, and beside the board a panel of buttons. A press of a button steps every
piece of its group to the next of its states: a tile opens or shuts, a bridge turns, a bar
grows or slides along a track. A key on the floor shows a new button when the avatar steps on
it; a press with the avatar on a pad of the button's group can take the avatar to another pad.
A press that takes the floor from under the avatar is undone, at a cost of steps.

From the frames alone:

* the avatar is the square of one colour that moves under the arrow keys (by its size; when
  several squares could be it, a first move shows which); the goal is a square of its size
  in a colour drawn nowhere else on the board, one that no button changes (a square reached
  without the level ending is not it);
* the board is the box of the background round the avatar; the panel is the rest of the
  screen between the board's rows, and its buttons are its solid shapes;
* a cell (the avatar's size, on its lattice) is floor when none of its pixels is of the
  background or of a colour a move was refused onto;
* a button's group is learned by pressing it until the board comes back: the pixels it
  changes, and their colours in each state of the cycle;
* a key is a small shape on floor that no button changes, drawn over the floor of another
  colour; a press that moves the avatar is a teleport, kept for that button and cell.

The plan: a breadth-first search over the avatar's cell and the state of each group, by moves
and presses, for the fewest actions onto the goal; a press that would leave the avatar off
the floor is not planned. With no plan, the nearest thing not known yet is tried: a key not
taken, or a press with the avatar on a piece of the button's group (once a piece).
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
MAX_FAILS = 10
MAX_LEVEL_ACTIONS = 300
MAX_STATES = 400_000
MAX_CYCLE = 12  # presses of one button before its cycle is taken not to close
MAX_SQUARE = 4  # the largest avatar looked for
MIN_BUTTON = 9  # a shape on the panel smaller than this is not taken for a button
MAX_ITEM = 16  # a shape on the floor larger than this is not taken for a key


# --- reading the screen ------------------------------------------------------------


def solid_squares(grid: np.ndarray, where: np.ndarray | None = None) -> list:
    """Squares of one colour drawn nowhere else (inside `where`, when given): [(x, y, s, colour)]."""
    g = grid.astype(np.int64) if where is None else np.where(where, grid, -1).astype(np.int64)
    counts = np.bincount(g[g >= 0].ravel(), minlength=16)
    out = []
    for c in np.nonzero((counts >= 4) & (counts <= MAX_SQUARE * MAX_SQUARE))[0]:
        ys, xs = np.nonzero(g == c)
        s = int(xs.max() - xs.min() + 1)
        if s == ys.max() - ys.min() + 1 and s * s == len(ys):
            out.append((int(xs.min()), int(ys.min()), s, int(c)))
    return out


def find_square(grid: np.ndarray, colour: int, s: int):
    """(x, y) of the one s x s square of the colour, or None."""
    found = [(c.x1, c.y1) for c in fp.segment(np.where(grid == colour, 1, 0))
             if c.color == 1 and c.width == s and c.height == s and c.is_rectangle]
    return found[0] if len(found) == 1 else None


def read_board(grid: np.ndarray, x: int, y: int, s: int, colour: int):
    """(background, (x0, y0, x1, y1)) of the board round the square at (x, y): the commonest
    colour near it, and the box of that colour's regions that reach near it (and of those that
    lie inside the box, until it holds)."""
    H, W = grid.shape
    r = 6 * s
    near = (slice(max(0, y - r), y + s + r), slice(max(0, x - r), x + s + r))
    cnt = Counter(int(v) for v in grid[near].ravel() if v != colour)
    if not cnt:
        return None
    bg = cnt.most_common(1)[0][0]
    comps = [c for c in fp.segment(grid) if c.color == bg]
    lo_x, hi_x, lo_y, hi_y = near[1].start, near[1].stop - 1, near[0].start, near[0].stop - 1
    box = None
    grown = True
    while grown:
        grown = False
        for c in comps:
            b = box or (lo_x, lo_y, hi_x, hi_y)
            if c.x2 < b[0] or c.x1 > b[2] or c.y2 < b[1] or c.y1 > b[3]:
                continue
            nb = (c.x1, c.y1, c.x2, c.y2) if box is None else (min(box[0], c.x1), min(box[1], c.y1), max(box[2], c.x2), max(box[3], c.y2))
            if nb != box:
                box, grown = nb, True
    if box is None or not (box[0] <= x and x + s - 1 <= box[2] and box[1] <= y and y + s - 1 <= box[3]):
        return None
    return bg, box


def read_buttons(grid: np.ndarray, box) -> list:
    """The solid shapes off the board: [(colour, (x1, y1, x2, y2), click point)]. The panel's
    own colour, the colour of the screen's margins and shapes on the screen's edge are left out."""
    H, W = grid.shape
    x0, y0, x1, y1 = box
    off = np.ones((H, W), dtype=bool)
    off[y0:y1 + 1, x0:x1 + 1] = False
    beside = grid[y0:y1 + 1][:, [i for i in range(W) if not x0 <= i <= x1]]
    margin = grid[[i for i in range(H) if not y0 <= i <= y1]]
    skip = set()
    for part in (beside, margin):
        if part.size:
            skip.add(Counter(int(v) for v in part.ravel()).most_common(1)[0][0])
    out = []
    for c in fp.segment(np.where(off, grid, -1).astype(np.int64)):
        if c.color < 0 or c.color in skip or c.area < MIN_BUTTON:
            continue
        if c.x1 == 0 or c.y1 == 0 or c.x2 == W - 1 or c.y2 == H - 1:
            continue
        if c.area < 0.5 * c.width * c.height:
            continue
        out.append((c.color, (c.x1, c.y1, c.x2, c.y2), c.click_point()))
    return out


class Group:
    """What one button does: the board's pixels it changes and their colours in each state."""

    def __init__(self, button, idx: np.ndarray, images: list, W: int) -> None:
        self.button = button
        self.idx = idx  # flat indices of the pixels the button changes
        self.images = images  # the colours at idx in each state, the state it was found in first
        self.n = len(images)
        # pieces: the 8-connected parts of the region, by pixel
        ys, xs = np.divmod(idx, W)
        piece = np.full(len(idx), -1)
        where = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
        n = 0
        for i in range(len(idx)):
            if piece[i] >= 0:
                continue
            stack = [i]
            piece[i] = n
            while stack:
                j = stack.pop()
                y, x = int(ys[j]), int(xs[j])
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        k = where.get((y + dy, x + dx))
                        if k is not None and piece[k] < 0:
                            piece[k] = n
                            stack.append(k)
            n += 1
        self.piece = dict(zip(idx.tolist(), piece.tolist()))  # flat index -> piece number


class Scene:
    """The level as read: the board, the avatar's lattice, the squares that could be the goal."""

    def __init__(self, grid: np.ndarray, avatar, look, bg: int, box) -> None:
        H, W = grid.shape
        self.H, self.W = H, W
        x, y = avatar
        colour, s = look
        self.s = s
        self.bg = bg
        self.box = box
        x0, y0, x1, y1 = box
        self.cells = [(cx, cy) for cy in range(y0 + (y - y0) % s, y1 - s + 2, s)
                      for cx in range(x0 + (x - x0) % s, x1 - s + 2, s)]
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.block = np.array([[(cy + j) * W + cx + i for j in range(s) for i in range(s)] for cx, cy in self.cells], dtype=np.int64)
        self.base = np.full(H * W, -1, dtype=np.int64)  # the board as last seen, -1 where not seen
        inside = np.zeros((H, W), dtype=bool)
        inside[y0:y1 + 1, x0:x1 + 1] = True
        self.inside = inside.ravel()
        self.goals: set = set()  # squares of the avatar's size in a colour of their own


# --- the solver ------------------------------------------------------------------


class PanelSolver:
    name = "panel"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.look: tuple | None = None  # (the avatar's colour, its size), kept for the game
        self.floor_seen: set = set()  # colours the avatar was seen standing on
        self.effects = 0  # presses seen to change the board, in the game
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.scene: Scene | None = None
        self.queue: list = []
        self.expect = None
        self.actions = 0
        self.fails = 0
        self.probe: list | None = None  # squares that could be the avatar, while a move shows which
        self.probe_dirs: list = []
        self.buttons: dict = {}  # (colour, box) -> {"at": click point, "visible", "group": index, "dead"}
        self.groups: list = []
        self.states: tuple = ()
        self.learning = None  # (button, images, cell) while a button is pressed round its cycle
        self.teleports: dict = {}  # (group, cell) -> where a press takes the avatar from there
        self.tried: set = set()  # (group, piece) pressed with the avatar on it
        self.items: dict = {}  # (colour, x, y) -> a key's pixels (flat), colour and cells
        self.visited: set = set()  # items stepped on
        self.not_floor: set = set()  # the background, and colours a move was refused onto
        self.stops: set = set()  # cells a move was refused onto, though their colours carried the avatar
        self.falls: set = set()  # (group, cell, states) whose press dropped the avatar
        self.learn_falls: set = set()  # (button, cell) whose press dropped the avatar while learning

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    def _fail(self, why: str) -> None:
        self.fails += 1
        self.stats[why] += 1
        self.queue = []

    # --- reading -----------------------------------------------------------------

    def _setup(self, grid: np.ndarray) -> bool:
        """Read the level; when two squares could be the avatar, start a probe instead."""
        if self.look is not None:
            at = find_square(grid, *self.look)
            if at is None:
                return False
            return self._scene(grid, at)
        cands = [c for c in solid_squares(grid) if c[2] >= 2]
        if not cands:
            return False
        if len(cands) == 1:
            x, y, s, colour = cands[0]
            self.found_look = (colour, s)
            return self._scene(grid, (x, y), (colour, s))
        # the squares that could move: the way most of them can move first
        bgs = [read_board(grid, x, y, s, colour) for x, y, s, colour in cands]
        dirs = []
        for k, d in KEYS.items():
            n = 0
            for (x, y, s, colour), rb in zip(cands, bgs):
                tx, ty = x + d[0] * s, y + d[1] * s
                if rb is None or not (0 <= tx and tx + s <= grid.shape[1] and 0 <= ty and ty + s <= grid.shape[0]):
                    continue
                if not (grid[ty:ty + s, tx:tx + s] == rb[0]).any():
                    n += 1
            dirs.append((-n, len(dirs), k))
        self.probe = cands
        self.probe_dirs = [k for n, _, k in sorted(dirs) if n < 0]
        return bool(self.probe_dirs)

    def _scene(self, grid: np.ndarray, at, look=None) -> bool:
        look = look or self.look
        colour, s = look
        rb = read_board(grid, at[0], at[1], s, colour)
        if rb is None:
            return False
        bg, box = rb
        sc = Scene(grid, at, look, bg, box)
        inside = sc.inside.reshape(grid.shape)
        goals = [(g[0], g[1]) for g in solid_squares(grid, inside) if g[2] == s and g[3] != colour and g[3] != bg]
        sc.goals = {g for g in goals if g in sc.index}
        if not sc.goals:
            return False
        buttons = read_buttons(grid, box)
        if not buttons:
            return False
        self.scene = sc
        self.found_look = look
        self.cell = at
        self.not_floor.add(bg)
        self.not_floor -= self.floor_seen
        self._see(grid)
        self._read_panel(grid)
        self._compile()
        return True

    def _see(self, grid: np.ndarray) -> None:
        """Keep what the board shows now, the avatar's cell aside."""
        sc = self.scene
        flat = grid.ravel().astype(np.int64)
        seen = sc.inside.copy()
        seen[sc.block[sc.index[self.cell]]] = False
        sc.base[seen] = flat[seen]

    def _read_panel(self, grid: np.ndarray) -> None:
        now = {(c, b): at for c, b, at in read_buttons(grid, self.scene.box)}
        for key, at in now.items():
            if key not in self.buttons:
                self.buttons[key] = {"at": at, "group": None, "dead": False}
                self.stats["buttons"] += 1
        for key, b in self.buttons.items():
            b["visible"] = key in now

    def _board(self, grid: np.ndarray) -> np.ndarray:
        """The board's pixels now (flat, -1 off the board and under the avatar)."""
        sc = self.scene
        flat = np.where(sc.inside, grid.ravel().astype(np.int64), -1)
        flat[sc.block[sc.index[self.cell]]] = -1
        return flat

    def _avatar_now(self, grid: np.ndarray):
        at = find_square(grid, *self.found_look)
        if at is None or at not in self.scene.index:
            return None
        return at

    # --- the model ----------------------------------------------------------------

    def _compile(self) -> None:
        """What is floor: per cell, the part no group changes, and each group's part by state."""
        sc = self.scene
        nf = np.array(sorted(self.not_floor), dtype=np.int64)
        owner = np.full(sc.H * sc.W, -1, dtype=np.int64)
        for gi, g in enumerate(self.groups):
            owner[g.idx] = gi
        blk = sc.block
        base_bad = np.isin(sc.base[blk], nf) & (owner[blk] < 0)
        self.static_ok = ~base_bad.any(axis=1)
        self.touching = [sorted({int(o) for o in owner[b] if o >= 0}) for b in blk]
        self.ok = []
        for gi, g in enumerate(self.groups):
            pos = {int(p): i for i, p in enumerate(g.idx)}
            per = []
            for img in g.images:
                bad = np.zeros(len(sc.cells), dtype=bool)
                for ci, b in enumerate(blk):
                    if gi not in self.touching[ci]:
                        continue
                    bad[ci] = any(img[pos[int(p)]] in self.not_floor for p in b if int(p) in pos)
                per.append(~bad)
            self.ok.append(per)
        self.floor_memo: dict = {}
        # a square some button changes is a piece, not the goal
        sc.goals = {g for g in sc.goals if not self.touching[sc.index[g]]}
        self._find_items()

    def _floor(self, cell, states: tuple) -> bool:
        ci = self.scene.index.get(cell)
        if ci is None or not self.static_ok[ci] or cell in self.stops:
            return False
        t = self.touching[ci]
        if not t:
            return True
        key = (ci, tuple(states[g] for g in t))
        r = self.floor_memo.get(key)
        if r is None:
            r = all(self.ok[g][states[g]][ci] for g in t)
            self.floor_memo[key] = r
        return r

    def _find_items(self) -> None:
        """Small shapes on floor no button changes, drawn over floor of another colour."""
        sc = self.scene
        W = sc.W
        owned = np.zeros(sc.H * sc.W, dtype=bool)
        for g in self.groups:
            owned[g.idx] = True
        img = np.where(sc.inside, sc.base, -1).reshape(sc.H, sc.W)
        items = {}
        for c in fp.segment(img, connectivity=8):
            if c.color < 0 or c.color in self.not_floor or c.area > MAX_ITEM:
                continue
            flat = c.pixels[:, 0] * W + c.pixels[:, 1]
            if owned[flat].any():
                continue
            cells = self._cells_over(flat)
            if not cells or not all(self.static_ok[sc.index[x]] for x in cells):
                continue
            other = {int(v) for x in cells for v in sc.base[sc.block[sc.index[x]]]} - {c.color, -1}
            if not other:
                continue
            key = (c.color, c.x1, c.y1)
            items[key] = {"pixels": flat, "colour": c.color, "cells": cells}
        self.items = {k: v for k, v in items.items() if not v["cells"] & sc.goals}

    def _cells_over(self, flat: np.ndarray) -> set:
        sc = self.scene
        s = sc.s
        ys, xs = np.divmod(flat, sc.W)
        x0, y0 = sc.cells[0]
        out = set()
        for y, x in zip(ys.tolist(), xs.tolist()):
            cx = x - (x - x0) % s
            cy = y - (y - y0) % s
            if (cx, cy) in sc.index:
                out.add((cx, cy))
            else:
                return set()
        return out

    def _colours(self, cell, states: tuple) -> set:
        """The colours of a cell by the model, in the given states."""
        sc = self.scene
        vals = sc.base[sc.block[sc.index[cell]]].copy()
        for gi, g in enumerate(self.groups):
            pos = {int(p): i for i, p in enumerate(g.idx)}
            for i, p in enumerate(sc.block[sc.index[cell]]):
                if int(p) in pos:
                    vals[i] = g.images[states[gi]][pos[int(p)]]
        return {int(v) for v in vals if v >= 0}

    def _pieces_at(self, gi: int, cell) -> set:
        g = self.groups[gi]
        sc = self.scene
        return {g.piece[int(p)] for p in sc.block[sc.index[cell]] if int(p) in g.piece}

    # --- planning -------------------------------------------------------------------

    def _press(self, key) -> Action:
        x, y = self.buttons[key]["at"]
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}

    def _next(self, st):
        cell, states = st
        s = self.scene.s
        for k, d in KEYS.items():
            c = (cell[0] + d[0] * s, cell[1] + d[1] * s)
            if self._floor(c, states):
                yield ("move", k, None), (c, states)
        for gi, g in enumerate(self.groups):
            if not self.buttons[g.button]["visible"]:
                continue
            ns = states[:gi] + ((states[gi] + 1) % g.n,) + states[gi + 1:]
            c = self.teleports.get((gi, cell), cell)
            if (gi, cell, states) not in self.falls and self._floor(c, ns):
                yield ("press", gi, None), (c, ns)

    def _bfs(self, target):
        """The fewest actions from now to a state (or a step) `target` accepts: [(step, state)]."""
        start = (self.cell, self.states)
        prev = {start: None}
        dq = deque([start])
        while dq:
            st = dq.popleft()
            hit = target(st, None, None)
            if hit:
                return self._unroll(prev, st, hit if hit is not True else None)
            for step, nst in self._next(st):
                hit = target(st, step, nst)
                if hit:
                    return self._unroll(prev, st, hit)
                if nst in prev:
                    continue
                prev[nst] = (st, step)
                dq.append(nst)
            if len(prev) > MAX_STATES:
                self.stats["search_cut"] += 1
                return None
        return None

    def _unroll(self, prev, st, last):
        out = []
        if last is not None:
            out.append(last)
        while prev[st] is not None:
            p, step = prev[st]
            out.append((step, st))
            st = p
        return out[::-1]

    def _plan(self):
        sc = self.scene
        # buttons not pressed yet are pressed round their cycle first, from floor no group changes
        todo = [k for k, b in self.buttons.items() if b["visible"] and b["group"] is None and not b["dead"]]
        if self.learning is not None:
            todo = [self.learning[0]]
        if todo:
            key = todo[0]

            def safe(st, step, nst):
                if step is not None:
                    return False
                cell, states = st
                return not self.touching[sc.index[cell]] and (key, cell) not in self.learn_falls

            path = self._bfs(safe)
            if path is None:
                self.buttons[key]["dead"] = True
                self.stats["unlearnable"] += 1
                return self._plan()
            return path + [(("learn", key, None), None)]
        if not self.groups:
            if not self.effects:
                self.disabled = True  # no press changed the board: not this family
            return None
        goal = self._bfs(lambda st, step, nst: step is None and st[0] in sc.goals)
        if goal is not None:
            return goal

        def unknown(st, step, nst):
            if step is None:
                return False
            if step[0] == "move":
                c = nst[0]
                for k, it in self.items.items():
                    if k not in self.visited and c in it["cells"]:
                        return (("move", step[1], k), nst)
                return False
            gi = step[1]
            cell, states = st
            if (gi, cell) in self.teleports:
                return False
            new = self._pieces_at(gi, cell) - {p for g, p in self.tried if g == gi}
            if new and nst[0] == cell:
                return (("press", gi, (gi, min(new))), nst)
            return False

        path = self._bfs(unknown)
        if path is not None:
            self.stats["explore"] += 1
        return path

    # --- acting ----------------------------------------------------------------------

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
            if self.probe is None:
                if self.read:
                    return None
                self.read = True
                if not self._setup(grid):
                    self.gave_up = True  # not this family's screen (looked at once a level)
                    return None
            if self.probe is not None:
                if not self.probe_dirs:
                    self.gave_up = True
                    return None
                k = self.probe_dirs.pop(0)
                self.expect = ("probe", grid, k)
                self.actions += 1
                self.stats["probe_moves"] += 1
                return k, None
        if not self.queue:
            plan = self._plan()
            if not plan:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            self.queue = plan
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        step, predicted = self.queue.pop(0)
        kind, what, extra = step
        self.expect = (kind, what, extra, self.cell, self.states, predicted, grid)
        self.actions += 1
        self.stats["actions"] += 1
        if kind == "move":
            return what, None
        if kind == "learn":
            if self.learning is None:
                self.learning = (what, [self._board(grid)], self.cell)
            return self._press(what)
        return self._press(self.groups[what].button)

    # --- observing --------------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
                self.look = self.found_look
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        exp, self.expect = self.expect, None
        if action is None or exp is None:
            return
        grid = np.asarray(obs.frame[-1])
        if action is GameAction.RESET:
            self.scene = None
            self.read = False
            self.queue = []
            return
        if exp[0] == "probe":
            self._observe_probe(exp, grid)
            return
        if self.scene is None:
            return
        kind, what, extra, cell0, states0, predicted, grid0 = exp
        now = self._avatar_now(grid)
        if now is None:
            self._fail("avatar_lost")
            self._give_up()
            return
        self.cell = now
        # a press that took the floor from under the avatar shows the board changed, then
        # the board as it was
        fell = False
        if kind in ("learn", "press") and len(obs.frame) > 1 and now == cell0:
            fell = (self._board(grid0) == self._board(grid)).all() and \
                any((np.asarray(f) != grid).any() for f in obs.frame[:-1])
        if kind == "move":
            self._observe_move(what, extra, cell0, now, predicted, grid0, grid)
        elif kind == "learn":
            self._observe_learn(what, cell0, now, fell, grid)
        else:
            self._observe_press(what, extra, cell0, states0, now, predicted, fell, grid)
        self._see(grid)
        shown = len(self.buttons)
        self._read_panel(grid)
        if kind == "move" and len(self.buttons) > shown:
            self.stats["keys_taken"] += 1  # a step showed a new button
        if any(b["visible"] and b["group"] is None and not b["dead"] for b in self.buttons.values()):
            self.queue = []  # a new button to learn first

    def _observe_probe(self, exp, grid: np.ndarray) -> None:
        _, grid0, k = exp
        d = KEYS[k]
        moved = []
        for x, y, s, colour in self.probe:
            at = find_square(grid, colour, s)
            if at == (x + d[0] * s, y + d[1] * s):
                moved.append((at, (colour, s)))
        if len(moved) != 1:
            if not self.probe_dirs:
                self.probe = None
                self.gave_up = True
            return
        at, look = moved[0]
        self.probe = None
        self.stats["probed"] += 1
        if not self._scene(grid, at, look):
            self.gave_up = True

    def _observe_move(self, k, item, cell0, now, predicted, grid0, grid) -> None:
        sc = self.scene
        target = predicted[0]
        if now == target:
            if now in sc.goals:
                sc.goals.discard(now)  # the level did not end here
                self.stats["not_goal"] += 1
                self.queue = []
            # the cell left is in sight now: its colours carry the avatar
            seen = {int(v) for v in grid.ravel()[sc.block[sc.index[cell0]]]}
            new = seen - self.floor_seen
            if new:
                self.floor_seen |= new
                if self.not_floor & new:
                    self.not_floor -= new
                    self._compile()
            if item is not None:
                self.visited.add(item)
            return
        if now == cell0:
            colours = {int(v) for v in grid0.ravel()[sc.block[sc.index[target]]]} - self.floor_seen
            if colours:
                self.not_floor |= colours
                self.stats["learned_not_floor"] += 1
            else:
                self.stops.add(target)
            self._compile()
            self._fail("refused")
            return
        self._fail("mismatch")

    def _observe_learn(self, key, cell0, now, fell, grid) -> None:
        if self.learning is None:
            return
        _, images, cell = self.learning
        if fell:
            self.learn_falls.add((key, cell0))
            self.learning = None
            self._fail("fell_learning")
            return
        if now != cell0:
            self.learning = None
            self.buttons[key]["dead"] = True
            self._fail("moved_learning")
            return
        img = self._board(grid)
        first = images[0]
        known = (img >= 0) & (first >= 0)
        if (img[known] == first[known]).all():
            self.learning = None
            b = self.buttons[key]
            if len(images) == 1:
                b["dead"] = True  # it changes nothing on the board
                self.stats["dead_buttons"] += 1
                return
            stack = np.stack(images)
            known_all = (stack >= 0).all(axis=0)
            changed = np.nonzero(known_all & (stack != stack[0]).any(axis=0))[0]
            g = Group(key, changed, [im[changed] for im in images], self.scene.W)
            b["group"] = len(self.groups)
            self.groups.append(g)
            self.states = self.states + (0,)
            self.effects += 1
            self.stats["learned_groups"] += 1
            self.stats[f"cycle_{g.n}"] += 1
            self._compile()
            return
        last = images[-1]
        known = (img >= 0) & (last >= 0)
        if len(images) > 1 and (img[known] == last[known]).all():
            # the press changed nothing this time: the button does not go round a cycle
            self.learning = None
            self.buttons[key]["dead"] = True
            self._fail("stuck")
            return
        images.append(img)
        if len(images) > MAX_CYCLE:
            self.learning = None
            self.buttons[key]["dead"] = True
            self._fail("no_cycle")

    def _observe_press(self, gi, explored, cell0, states0, now, predicted, fell, grid) -> None:
        if explored is not None:
            self.tried.add(explored)
        if fell:
            # the press was undone: some colour the avatar was left on does not carry it
            colours = self._colours(*predicted) - self.floor_seen
            if colours:
                self.not_floor |= colours
                self.stats["learned_not_floor"] += 1
                self._compile()
            self.falls.add((gi, cell0, states0))
            self._fail("fell")
            return
        states = self._states_now(grid, predicted[1])
        if states is None:
            self._fail("board_mismatch")
            states = predicted[1]
        self.states = states
        if now != predicted[0]:
            if now != cell0:
                self.teleports[(gi, cell0)] = now
                self.stats["learned_teleport"] += 1
                # the way back is taken to be the same press from the pad it led to
                self.teleports.setdefault((gi, now), cell0)
                self.tried |= {(gi, p) for p in self._pieces_at(gi, now)}
            else:
                self.teleports.pop((gi, cell0), None)
            self.queue = []
        elif states != predicted[1]:
            self.queue = []

    def _states_now(self, grid: np.ndarray, guess: tuple):
        """Each group's state as the board shows it (the guess where the board fits several)."""
        img = self._board(grid)
        out = []
        for gi, g in enumerate(self.groups):
            vals = img[g.idx]
            known = vals >= 0
            fit = [j for j, im in enumerate(g.images) if (im[known] == vals[known]).all()]
            if not fit:
                return None
            out.append(guess[gi] if guess[gi] in fit else fit[0])
        return tuple(out)
