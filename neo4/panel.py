"""The panel family: buttons on a panel beside the board step groups of floor pieces through
their states, or drive a piece about the board, and the avatar walks over the floor to the goal.

Some puzzles show a board of floor pieces on a background, an avatar that walks on the floor
only, a goal square, and beside the board a panel of buttons. A press of a button steps every
piece of its group to the next of its states: a tile opens or shuts, a bridge turns, a bar
grows or slides along a track. Other buttons drive a piece (a crane) a step at a time along a
track, and one makes it grip the piece beside it, which it then carries. A key on the floor
shows a new button when the avatar steps on it; a press with the avatar on a pad of the
button's group can take the avatar to another pad. A press that takes the floor from under the
avatar is undone, at a cost of steps.

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
  changes, and their colours in each state of the cycle. A button whose presses move one
  piece by the same step until a press moves nothing drives that piece; a press that
  moves nothing is the end of its track, and the step is taken to go on elsewhere until a
  press there is seen not to. A press that changes nothing is tried again with the piece
  elsewhere; one that changes the driven piece where it stands made it grip, and the next
  step shows what it carries;
* a key is a small shape on floor that no button changes, drawn over the floor of another
  colour; a press that moves the avatar is a teleport, kept for that button and cell.

The plan: a breadth-first search over the avatar's cell, the state of each group and where
the driven piece is (and whether it holds), by moves and presses, for the fewest actions onto
the goal; a press that would leave the avatar off the floor is not planned, and a group is
pressed only with the avatar near its pieces (then anywhere, when that finds nothing). With no
plan, the nearest thing not known yet is tried: a key not taken, a press with the avatar on a
piece of the button's group (once a piece), a press that did nothing with the driven piece
elsewhere, or a step of a piece that holds something not seen carried yet.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
MAX_FAILS = 10
MAX_LEVEL_ACTIONS = 500
MAX_STATES = 400_000
MAX_CYCLE = 12  # presses of one button before its cycle is taken not to close
MAX_SQUARE = 4  # the largest avatar looked for
MIN_BUTTON = 9  # a shape on the panel smaller than this is not taken for a button
MAX_ITEM = 16  # a shape on the floor larger than this is not taken for a key
MAX_BODY = 200  # a shape larger than this is not taken for a piece that a press moves
MAX_STEP = 8  # the longest step looked for in a piece a press moves
MIN_PIECE = 4  # a moved shape smaller than this beside a larger one is a speck of ground showing through
NEAR = 2  # a group is pressed only with the avatar this many cells from its pieces, at first
UNSEEN = -2  # the ground under a driven piece, never seen uncovered


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


def moved(b0: np.ndarray, b1: np.ndarray, H: int, W: int, bg: int):
    """((dx, dy), bodies) when all that changed between two boards (flat, -1 where unknown) is
    pieces moved by one step, each body {flat index: colour} as it lay before, and every pixel
    a body left now shows something else; else None. The background is no piece."""
    changed = (b0 >= 0) & (b1 >= 0) & (b0 != b1)
    if not changed.any():
        return None
    cands = []
    for c in fp.segment(b0.reshape(H, W), connectivity=8):
        if c.color < 0 or c.color == bg or c.area > MAX_BODY:
            continue
        flat = c.pixels[:, 0] * W + c.pixels[:, 1]
        if changed[flat].any():
            cands.append((c, flat))
    for dist in range(1, MAX_STEP + 1):  # the shortest step first
        for dy in range(-dist, dist + 1):
            for dx in range(-dist, dist + 1):
                if max(abs(dx), abs(dy)) != dist:
                    continue
                cover = np.zeros(H * W, dtype=bool)
                bodies = []
                for c, flat in cands:
                    ys, xs = c.pixels[:, 0] + dy, c.pixels[:, 1] + dx
                    if ys.min() < 0 or xs.min() < 0 or ys.max() >= H or xs.max() >= W:
                        continue
                    to = ys * W + xs
                    vals = b1[to]
                    if not ((vals == c.color) | (vals < 0)).all():
                        continue
                    left = np.setdiff1d(flat, to)
                    left = left[b1[left] >= 0]
                    if not changed[left].all():
                        continue  # what it left still shows it: it did not move
                    bodies.append((flat, to, {int(p): c.color for p in flat}))
                if any(len(f) >= MIN_PIECE for f, _, _ in bodies):
                    bodies = [b for b in bodies if len(b[0]) >= MIN_PIECE]  # specks of the ground beside it aside
                for flat, to, _ in bodies:
                    cover[flat] = True
                    cover[to] = True
                if bodies and not (changed & ~cover).any():
                    return (dx, dy), [bd for _, _, bd in bodies]
    return None


def shift(px: dict, dx: int, dy: int, H: int, W: int):
    """The pixels moved by (dx, dy), or None when that takes one off the screen."""
    out = {}
    for p, c in px.items():
        y, x = divmod(p, W)
        x, y = x + dx, y + dy
        if not (0 <= x < W and 0 <= y < H):
            return None
        out[y * W + x] = c
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


class Mover:
    """A piece that some buttons step about the board, each by its own step, and that one
    button makes grip the piece beside it, which it then carries. Where it is, is its offset
    from where it was first seen."""

    def __init__(self, body: dict) -> None:
        self.body = body  # flat index -> colour, as first seen (offset (0, 0))
        self.held = None  # its look while it holds something, at offset (0, 0)
        self.steps: dict = {}  # button -> (dx, dy)
        self.blocked: set = set()  # (dx, dy, button): a press there moved nothing
        self.grip = None  # the button that grips
        self.grip_at = None  # the offset at which it took hold
        self.cargo = None  # flat index -> colour of what it carries, as it lay when gripped


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
        self.ground = self.base  # the board without the driven piece and what it carries
        inside = np.zeros((H, W), dtype=bool)
        inside[y0:y1 + 1, x0:x1 + 1] = True
        self.inside = inside.ravel()
        self.goals: set = set()  # squares of the avatar's size in a colour of their own

    def cell_of(self, p: int):
        """The lattice cell a pixel lies in, or None."""
        y, x = divmod(p, self.W)
        x0, y0 = self.cells[0]
        c = (x - (x - x0) % self.s, y - (y - y0) % self.s)
        return c if c in self.index else None


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
        # (colour, box) -> {"at": click point, "visible", "group": index, "role": "step" | "grip",
        # "dead", "idle": the driven piece's offsets at which a press changed nothing}
        self.buttons: dict = {}
        self.groups: list = []
        self.states: tuple = ()
        self.mover: Mover | None = None
        self.mstate = None  # (dx, dy, holds) of the driven piece
        self.learning = None  # (button, images, cell) while a button is pressed round its cycle
        self.teleports: dict = {}  # (group, cell) -> where a press takes the avatar from there
        self.tried: set = set()  # (group, piece) pressed with the avatar on it
        self.items: dict = {}  # (colour, x, y) -> a key's pixels (flat), colour and cells
        self.visited: set = set()  # items stepped on
        self.not_floor: set = set()  # the background, and colours a move was refused onto
        self.stops: set = set()  # cells a move was refused onto, though their colours carried the avatar
        self.falls: set = set()  # (button, cell, states, piece's state) whose press dropped the avatar
        self.learn_falls: set = set()  # (button, cell) whose press dropped the avatar while learning
        self.ov_cache: dict = {}

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
        """Keep what the board shows now, the avatar's cell aside (and, for the ground, the
        driven piece and what it carries)."""
        sc = self.scene
        flat = grid.ravel().astype(np.int64)
        seen = sc.inside.copy()
        seen[sc.block[sc.index[self.cell]]] = False
        sc.base[seen] = flat[seen]
        if self.mover is not None:
            ov = self._overlay(self.mstate)
            if ov is not None:
                seen[list(ov[0])] = False
            revealed = seen & (sc.ground == UNSEEN)
            sc.ground[seen] = flat[seen]
            if revealed.any():
                self._compile()

    def _read_panel(self, grid: np.ndarray) -> None:
        now = {(c, b): at for c, b, at in read_buttons(grid, self.scene.box)}
        for key, at in now.items():
            if key not in self.buttons:
                self.buttons[key] = {"at": at, "group": None, "role": None, "dead": False, "idle": set()}
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
        ground = sc.ground[blk]
        base_bad = (np.isin(ground, nf) | (ground == UNSEEN)) & (owner[blk] < 0)
        self.static_ok = ~base_bad.any(axis=1)
        self.touching = [sorted({int(o) for o in owner[b] if o >= 0}) for b in blk]
        self.ok = []
        self.near = []
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
            near = set()
            for ci in range(len(sc.cells)):
                if gi in self.touching[ci]:
                    cx, cy = sc.cells[ci]
                    near |= {(cx + i * sc.s, cy + j * sc.s) for i in range(-NEAR, NEAR + 1) for j in range(-NEAR, NEAR + 1)}
            self.near.append(near)
        self.floor_memo: dict = {}
        self.ov_cache = {}
        # a square some button changes is a piece, not the goal
        sc.goals = {g for g in sc.goals if not self.touching[sc.index[g]]}
        self._find_items()

    def _overlay(self, m):
        """The driven piece and what it carries in state m: (pixels {flat: colour}, the driven
        piece's own pixels, the cells they touch); None when that is off the board."""
        r = self.ov_cache.get(m, False)
        if r is not False:
            return r
        sc = self.scene
        mv = self.mover
        dx, dy, holds = m
        px: dict = {}
        r = None
        ok = True
        if mv.cargo is not None:
            gx, gy = mv.grip_at
            cargo = shift(mv.cargo, dx - gx, dy - gy, sc.H, sc.W) if holds else mv.cargo
            if cargo is None:
                ok = False
            else:
                px.update(cargo)
        body = shift(mv.held if holds and mv.held else mv.body, dx, dy, sc.H, sc.W)
        if ok and body is not None:
            px.update(body)
            if all(sc.inside[p] for p in px):
                cells = {c for c in map(sc.cell_of, px) if c is not None}
                r = (px, set(body), cells)
        self.ov_cache[m] = r
        return r

    def _floor(self, cell, states: tuple, m=None) -> bool:
        ci = self.scene.index.get(cell)
        if ci is None or cell in self.stops:
            return False
        t = self.touching[ci]
        if m is not None:
            ov = self._overlay(m)
            if ov is not None and cell in ov[2]:
                key = (ci, m, tuple(states[g] for g in t))
                r = self.floor_memo.get(key)
                if r is None:
                    r = self._composed_ok(ci, states, ov)
                    self.floor_memo[key] = r
                return r
        if not self.static_ok[ci]:
            return False
        if not t:
            return True
        key = (ci, tuple(states[g] for g in t))
        r = self.floor_memo.get(key)
        if r is None:
            r = all(self.ok[g][states[g]][ci] for g in t)
            self.floor_memo[key] = r
        return r

    def _composed_ok(self, ci: int, states: tuple, ov) -> bool:
        """A cell the driven piece or its load touches: not on the piece itself, and every
        pixel of a colour that carries the avatar."""
        px, body, _ = ov
        for p in self.scene.block[ci].tolist():
            if p in body:
                return False
            v = px.get(p)
            if v is None:
                v = self._pixel(p, states)
            if v == UNSEEN or v in self.not_floor:
                return False
        return True

    def _pixel(self, p: int, states: tuple) -> int:
        """A pixel's colour by the groups' states, else as the ground shows it."""
        for gi, g in enumerate(self.groups):
            if p in g.piece:
                return int(g.images[states[gi]][np.searchsorted(g.idx, p)])
        return int(self.scene.ground[p])

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
        out = set()
        for p in flat.tolist():
            c = self.scene.cell_of(p)
            if c is None:
                return set()
            out.add(c)
        return out

    def _colours(self, cell, states: tuple, m=None) -> set:
        """The colours of a cell by the model, in the given states."""
        sc = self.scene
        ov = self._overlay(m) if m is not None else None
        out = set()
        for p in sc.block[sc.index[cell]].tolist():
            v = ov[0].get(p) if ov is not None else None
            if v is None:
                v = self._pixel(p, states)
            if v >= 0:
                out.add(v)
        return out

    def _pieces_at(self, gi: int, cell) -> set:
        g = self.groups[gi]
        sc = self.scene
        return {g.piece[int(p)] for p in sc.block[sc.index[cell]] if int(p) in g.piece}

    # --- planning -------------------------------------------------------------------

    def _press(self, key) -> Action:
        x, y = self.buttons[key]["at"]
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}

    def _next(self, st, prune: bool):
        cell, states, m = st
        s = self.scene.s
        for k, d in KEYS.items():
            c = (cell[0] + d[0] * s, cell[1] + d[1] * s)
            if self._floor(c, states, m):
                yield ("move", k, None), (c, states, m)
        for gi, g in enumerate(self.groups):
            if not self.buttons[g.button]["visible"] or (prune and cell not in self.near[gi]):
                continue
            ns = states[:gi] + ((states[gi] + 1) % g.n,) + states[gi + 1:]
            c = self.teleports.get((gi, cell), cell)
            if (g.button, cell, states, m) not in self.falls and self._floor(c, ns, m):
                yield ("press", g.button, None), (c, ns, m)
        mv = self.mover
        if mv is None:
            return
        dx, dy, holds = m
        for key, (sx, sy) in mv.steps.items():
            if not self.buttons[key]["visible"] or (dx, dy, key) in mv.blocked:
                continue
            nm = (dx + sx, dy + sy, holds)
            if self._overlay(nm) is not None and (key, cell, states, m) not in self.falls and self._floor(cell, states, nm):
                yield ("press", key, None), (cell, states, nm)
        if mv.grip is not None and not holds and mv.grip_at == (dx, dy) and self.buttons[mv.grip]["visible"]:
            nm = (dx, dy, 1)
            if self._overlay(nm) is not None and self._floor(cell, states, nm):
                yield ("press", mv.grip, None), (cell, states, nm)

    def _bfs(self, target, prune: bool = True):
        """The fewest actions from now to a state (or a step) `target` accepts: [(step, state)]."""
        start = (self.cell, self.states, self.mstate)
        prev = {start: None}
        dq = deque([start])
        while dq:
            st = dq.popleft()
            hit = target(st, None, None)
            if hit:
                return self._unroll(prev, st, hit if hit is not True else None)
            for step, nst in self._next(st, prune):
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
        todo = [k for k, b in self.buttons.items() if b["visible"] and b["group"] is None and b["role"] is None
                and not b["dead"] and not b["idle"]]
        if self.learning is not None:
            todo = [self.learning[0]]
        if todo:
            key = todo[0]

            def safe(st, step, nst):
                if step is not None:
                    return False
                cell, states, m = st
                if m is not None and cell in (self._overlay(m) or ((), (), ()))[2]:
                    return False
                return not self.touching[sc.index[cell]] and (key, cell) not in self.learn_falls

            path = self._bfs(safe)
            if path is None:
                self.buttons[key]["dead"] = True
                self.stats["unlearnable"] += 1
                return self._plan()
            return path + [(("learn", key, None), None)]
        if not self.groups and self.mover is None:
            if not self.effects:
                self.disabled = True  # no press changed the board: not this family
            return None

        def unknown(st, step, nst):
            cell, states, m = st
            if step is None:
                # a button that did nothing, pressed again with the driven piece elsewhere
                if m is None:
                    return False
                for key, b in self.buttons.items():
                    if b["visible"] and b["idle"] and b["group"] is None and b["role"] is None and not b["dead"] \
                            and (m[0], m[1]) not in b["idle"]:
                        return (("press", key, ("idle", key)), st)
                return False
            if step[0] == "move":
                c = nst[0]
                for k, it in self.items.items():
                    if k not in self.visited and c in it["cells"]:
                        return (("move", step[1], k), nst)
                return False
            b = self.buttons[step[1]]
            if b["role"] == "step" and m[2] and self.mover.cargo is None:
                return (("press", step[1], ("carry", None)), nst)  # to see what it carries
            gi = b["group"]
            if gi is None or (gi, cell) in self.teleports:
                return False
            new = self._pieces_at(gi, cell) - {p for g, p in self.tried if g == gi}
            if new and nst[0] == cell:
                return (("press", step[1], ("piece", (gi, min(new)))), nst)
            return False

        for prune in (True, False):
            goal = self._bfs(lambda st, step, nst: step is None and st[0] in sc.goals, prune)
            if goal is not None:
                return goal
            path = self._bfs(unknown, prune)
            if path is not None:
                self.stats["explore"] += 1
                return path
        return None

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
        self.expect = (kind, what, extra, self.cell, self.states, self.mstate, predicted, grid)
        self.actions += 1
        self.stats["actions"] += 1
        if kind == "move":
            return what, None
        if kind == "learn" and self.learning is None:
            self.learning = (what, [self._board(grid)], self.cell)
        return self._press(what)

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
        kind, what, extra, cell0, states0, m0, predicted, grid0 = exp
        now = self._avatar_now(grid)
        if now is None:
            self._fail("avatar_lost")
            self._give_up()
            return
        b0 = self._board(grid0)  # the board before, with the avatar where it stood
        self.cell = now
        # a press that took the floor from under the avatar shows the avatar going, then the
        # board as it was; a press that moved nothing may show a piece bumping, the avatar still
        fell = False
        if kind in ("learn", "press") and len(obs.frame) > 1 and now == cell0:
            colour = self.found_look[0]
            blk = self.scene.block[self.scene.index[cell0]]
            fell = (b0 == self._board(grid)).all() and \
                any((np.asarray(f).ravel()[blk] != colour).any() for f in obs.frame[:-1])
        if kind == "move":
            self._observe_move(what, extra, cell0, now, predicted, grid0, grid)
        elif kind == "learn":
            self._observe_learn(what, cell0, now, fell, b0, grid)
        else:
            self._observe_press(what, extra, cell0, states0, m0, now, predicted, fell, b0, grid)
        self._see(grid)
        shown = len(self.buttons)
        self._read_panel(grid)
        if kind == "move" and len(self.buttons) > shown:
            self.stats["keys_taken"] += 1  # a step showed a new button
        if any(b["visible"] and b["group"] is None and b["role"] is None and not b["dead"] and not b["idle"]
               for b in self.buttons.values()):
            self.queue = []  # a new button to learn first

    def _observe_probe(self, exp, grid: np.ndarray) -> None:
        _, grid0, k = exp
        d = KEYS[k]
        moved_to = []
        for x, y, s, colour in self.probe:
            at = find_square(grid, colour, s)
            if at == (x + d[0] * s, y + d[1] * s):
                moved_to.append((at, (colour, s)))
        if len(moved_to) != 1:
            if not self.probe_dirs:
                self.probe = None
                self.gave_up = True
            return
        at, look = moved_to[0]
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

    def _observe_learn(self, key, cell0, now, fell, b0, grid) -> None:
        if self.learning is None:
            return
        _, images, cell = self.learning
        b = self.buttons[key]
        if fell:
            self.learn_falls.add((key, cell0))
            self.learning = None
            self._fail("fell_learning")
            return
        if now != cell0:
            self.learning = None
            b["dead"] = True
            self._fail("moved_learning")
            return
        img = self._board(grid)
        first = images[0]
        known = (img >= 0) & (first >= 0)
        if (img[known] == first[known]).all():
            self.learning = None
            if len(images) == 1:
                # it changes nothing here; with a driven piece about, it is tried again with the
                # piece elsewhere
                b["idle"].add(self.mstate[:2] if self.mstate is not None else None)
                self.stats["idle_presses"] += 1
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
        if len(images) == 1 and self.mover is not None:
            # with a driven piece known, a press that steps it is one of its steps, and one
            # that changes it where it stands is its grip
            sc = self.scene
            t = moved(last, img, sc.H, sc.W, sc.bg)
            if t is not None and self._learn_step(key, t[0], t[1], 1, [last, img], stuck=False):
                self.learning = None
                return
            if self._in_place(last, img):
                self.learning = None
                self._learn_grip(key, img)
                return
        if len(images) > 1 and (img[known] == last[known]).all():
            # the press changed nothing this time: a piece moved by the same step each time has
            # come to the end of its track; else the button does not go round a cycle
            self.learning = None
            sc = self.scene
            trans = [moved(a, c, sc.H, sc.W, sc.bg) for a, c in zip(images, images[1:])]
            if all(t is not None for t in trans) and len({t[0] for t in trans}) == 1:
                if self._learn_step(key, trans[0][0], trans[0][1], len(trans), images, stuck=True):
                    return
            b["dead"] = True
            self._fail("stuck")
            return
        images.append(img)
        if len(images) > MAX_CYCLE:
            self.learning = None
            b["dead"] = True
            self._fail("no_cycle")

    def _in_place(self, b0: np.ndarray, b1: np.ndarray) -> bool:
        """Whether all a press changed lies on the driven piece where it stands."""
        ov = self._overlay(self.mstate)
        if ov is None:
            return False
        sc = self.scene
        changed = np.nonzero((b0 >= 0) & (b1 >= 0) & (b0 != b1))[0]
        if not len(changed):
            return False
        ys, xs = np.divmod(np.array(sorted(ov[1])), sc.W)
        cy, cx = np.divmod(changed, sc.W)
        return bool((cx >= xs.min() - 1).all() and (cx <= xs.max() + 1).all() and (cy >= ys.min() - 1).all() and (cy <= ys.max() + 1).all())

    def _learn_step(self, key, d, bodies, n: int, images: list, stuck: bool) -> bool:
        """A button that moved a piece by step d, n times (and then no further, when stuck): the
        driven piece, found or new."""
        sc = self.scene
        body = {}
        for bd in bodies:
            body.update(bd)
        b = self.buttons[key]
        if self.mover is None:
            mv = self.mover = Mover(body)
            self.mstate = (d[0] * n, d[1] * n, 0)
            for other in self.buttons.values():
                if None in other["idle"]:
                    other["idle"] = (other["idle"] - {None}) | {(0, 0)}
            # the ground under the piece: where some board seen showed it uncovered
            ground = sc.base.copy()
            now = shift(body, d[0] * n, d[1] * n, sc.H, sc.W) or {}
            for p in now:
                ground[p] = UNSEEN
            for i in range(len(images) - 1, -1, -1):
                at = shift(body, d[0] * i, d[1] * i, sc.H, sc.W) or {}
                for p in now:
                    if ground[p] == UNSEEN and p not in at and images[i][p] >= 0:
                        ground[p] = images[i][p]
            sc.ground = ground
            self.stats["movers"] += 1
        else:
            mv = self.mover
            dx, dy, holds = self.mstate  # where it was before these presses
            look = shift(mv.held if holds and mv.held else mv.body, dx, dy, sc.H, sc.W) or {}
            if not set(look) <= set(body):
                return False  # another piece: not driven by the same buttons
            self.mstate = (dx + d[0] * n, dy + d[1] * n, holds)
        mv.steps[key] = d
        if stuck:
            dx, dy, _ = self.mstate
            mv.blocked.add((dx, dy, key))
        b["role"] = "step"
        b["idle"] = set()
        self.effects += 1
        self.stats["learned_steps"] += 1
        self._compile()
        return True

    def _learn_cargo(self, bodies, m0) -> None:
        """The first step of a piece that holds something: what moved besides it is what it carries."""
        mv = self.mover
        dx, dy, holds = m0
        if not holds or mv.cargo is not None:
            return
        sc = self.scene
        look = set(shift(mv.held if mv.held else mv.body, dx, dy, sc.H, sc.W) or {})
        cargo = {}
        for bd in bodies:
            if not set(bd) & look:
                cargo.update(bd)
        if cargo:
            gx, gy = mv.grip_at
            mv.cargo = shift(cargo, gx - dx, gy - dy, sc.H, sc.W)
            self.stats["learned_cargo"] += 1
            self._compile()

    def _learn_grip(self, key, b1: np.ndarray) -> None:
        """A press that changed the driven piece where it stands: it grips there."""
        mv = self.mover
        sc = self.scene
        dx, dy, _ = self.mstate
        colours = set(mv.body.values()) - {sc.bg}
        ys, xs = np.divmod(np.array(sorted(mv.body)), sc.W)
        held = {}
        for y in range(ys.min() + dy - 1, ys.max() + dy + 2):
            for x in range(xs.min() + dx - 1, xs.max() + dx + 2):
                p = y * sc.W + x
                if 0 <= x < sc.W and 0 <= y < sc.H and b1[p] in colours:
                    held[(y - dy) * sc.W + x - dx] = int(b1[p])
        mv.held = held
        mv.grip = key
        mv.grip_at = (dx, dy)
        self.mstate = (dx, dy, 1)
        b = self.buttons[key]
        b["role"] = "grip"
        b["idle"] = set()
        self.effects += 1
        self.stats["learned_grip"] += 1
        self._compile()

    def _observe_press(self, key, extra, cell0, states0, m0, now, predicted, fell, b0, grid) -> None:
        b = self.buttons[key]
        why = extra[0] if extra else None
        if why == "piece":
            self.tried.add(extra[1])
        if fell:
            # the press was undone: some colour the avatar was left on does not carry it
            colours = self._colours(predicted[0], predicted[1], predicted[2]) - self.floor_seen
            if colours:
                self.not_floor |= colours
                self.stats["learned_not_floor"] += 1
                self._compile()
            self.falls.add((key, cell0, states0, m0))
            self._fail("fell")
            return
        b1 = self._board(grid)
        sc = self.scene
        if why == "idle":
            # a button that did nothing, pressed with the driven piece elsewhere
            t = moved(b0, b1, sc.H, sc.W, sc.bg)
            if not ((b0 >= 0) & (b1 >= 0) & (b0 != b1)).any():
                b["idle"].add(self.mstate[:2])
            elif t is not None and self._learn_step(key, t[0], t[1], 1, [b0, b1], stuck=False):
                self._learn_cargo(t[1], m0)
            elif self.mover is not None and self._in_place(b0, b1):
                self._learn_grip(key, b1)
            else:
                self.learning = (key, [b0, b1], cell0)  # it changes the board: round its cycle
            self.queue = []
            return
        if b["role"] == "step":
            mv = self.mover
            dx, dy, holds = m0
            t = moved(b0, b1, sc.H, sc.W, sc.bg)
            if not ((b0 >= 0) & (b1 >= 0) & (b0 != b1)).any():
                mv.blocked.add((dx, dy, key))  # the end of the track here
                self.stats["bumps"] += 1
                self.queue = []
                return
            if t is None or t[0] != mv.steps[key]:
                self._fail("mover_mismatch")
                return
            self._learn_cargo(t[1], m0)
            self.mstate = (dx + t[0][0], dy + t[0][1], holds)
            self._compile()
            if self.mstate != predicted[2]:
                self.queue = []
            return
        if b["role"] == "grip":
            if self._in_place(b0, b1) or self.mstate[2]:
                self.mstate = predicted[2]
            else:
                self._fail("grip_mismatch")
            return
        gi = b["group"]
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
        if self.mover is not None:
            ov = self._overlay(self.mstate)
            if ov is not None:
                img[list(ov[0])] = -1
        out = []
        for gi, g in enumerate(self.groups):
            vals = img[g.idx]
            known = vals >= 0
            fit = [j for j, im in enumerate(g.images) if (im[known] == vals[known]).all()]
            if not fit:
                return None
            out.append(guess[gi] if guess[gi] in fit else fit[0])
        return tuple(out)
