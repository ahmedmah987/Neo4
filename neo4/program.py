"""The program family: fill the slots of a program so that running it writes the target.

Some puzzles show a target sequence of colours and a small program to produce it:
frames (bordered boxes) hold a row of slots; a palette holds tiles, solid tiles of a
colour and hollow tiles of a frame's colour. Clicking a tile and then an empty slot
puts the tile there; one action runs the program. Running reads the slots of the
first frame in order: a solid tile writes its colour, which must be the next colour
of the target; a hollow tile runs the frame of its colour and comes back; an empty
slot, a wrong colour, or the first frame ending before the target is complete stops
the run. The target complete, the level is done.

From the frames alone:

* the target is a row (or rows, read in order) of equal hollow squares, outside every
  frame;
* a frame is a rectangle of one colour (two long runs of it, one above the other); its
  slots are what lies inside, left to right: small solid squares where a slot is
  empty, tiles where one is filled;
* the palette is the tiles outside every frame;
* the first frame is the topmost.

The solver searches the program's run, choosing a palette tile for each empty slot
the run reaches (the colour the target wants next, or a call), with the fewest tiles
placed, places them two clicks each and runs. A run that does not complete the
level, or a click that does not do what was planned, gives the level back.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp
from .marker import Comp, components

Action = tuple[GameAction, dict[str, int] | None]

RUN = GameAction.ACTION5
MIN_FRAME = 10  # a frame's border runs at least this long
MAX_NODES = 20000  # search steps per plan
MAX_STEPS = 400  # steps of one simulated run (a run that loops is a failure)
MAX_FAILS = 3  # clicks that did not do what was planned, per level


@dataclass(frozen=True)
class Cell:
    """What a slot or a palette place holds: None (empty), or a tile."""

    kind: str  # "empty", "tile" (solid) or "call" (hollow)
    color: int
    at: tuple[int, int]  # a click point (x, y)


@dataclass
class Frame:
    color: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 of the border
    slots: list[Cell]


def _square(c: Comp) -> str | None:
    """'solid' or 'hollow' for a square component, else None."""
    if c.w != c.h:
        return None
    n = c.w
    if c.area == n * n:
        return "solid"
    if n >= 3 and c.area == 4 * n - 4:
        inner = {(dy, dx) for dy in range(1, n - 1) for dx in range(1, n - 1)}
        if not (inner & c.rel):
            return "hollow"
    return None


def _centre(c: Comp) -> tuple[int, int]:
    return (c.x + c.w // 2, c.y + c.h // 2)


def _cell_at(grid: np.ndarray, bg: int, cx: int, cy: int) -> Cell | None:
    """A tile (a solid 4x4 square), a call (a 4x4 ring round the background) or an
    empty slot (a 2x2 square on the background) centred at (cx, cy)."""
    h, w = grid.shape
    if not (3 <= cx < w - 3 and 3 <= cy < h - 3):
        return None
    block = grid[cy - 2:cy + 2, cx - 2:cx + 2]
    col = int(block[0, 0])
    outer = grid[cy - 3:cy + 3, cx - 3:cx + 3].copy()
    outer[1:5, 1:5] = bg
    if col != bg and (outer != bg).sum() <= 2:  # on the background (a line may touch it)
        ring = np.ones((4, 4), dtype=bool)
        ring[1:3, 1:3] = False
        if (block == col).all():
            return Cell("tile", col, (cx, cy))
        if (block[ring] == col).all() and (block[1:3, 1:3] == bg).all():
            return Cell("call", col, (cx, cy))
    mark = grid[cy - 1:cy + 1, cx - 1:cx + 1]
    col = int(mark[0, 0])
    if col != bg and (mark == col).all():
        around = grid[cy - 2:cy + 2, cx - 2:cx + 2].copy()
        around[1:3, 1:3] = bg
        if (around == bg).all():
            return Cell("empty", col, (cx, cy))
    return None


def _cells(grid: np.ndarray, bg: int, box) -> list[Cell]:
    """The slots inside a frame, left to right, read by their pixels (a tile may be
    drawn touching a line of its colour)."""
    x1, y1, x2, y2 = box
    found: list[Cell] = []
    for cy in range(y1 + 3, y2 - 1):
        for cx in range(x1 + 3, x2 - 1):
            c = _cell_at(grid, bg, cx, cy)
            if c is not None and all(abs(c.at[0] - o.at[0]) >= 4 or abs(c.at[1] - o.at[1]) >= 4 for o in found):
                found.append(c)
    return sorted(found, key=lambda c: c.at[0])


def read_screen(grid: np.ndarray, mask) -> tuple[list[int], list[Frame], list[Cell]] | None:
    """The target, the frames and the palette, or None if the screen is not a program."""
    bg = int(np.bincount(grid.ravel()).argmax())
    comps = components(grid, mask)
    h, w = grid.shape
    # frames: a long horizontal run of one colour with another of the same colour
    # and extent below it
    runs: dict[int, list[tuple[int, int, int]]] = {}
    for y in range(h):
        row = grid[y]
        x = 0
        while x < w:
            col = int(row[x])
            x2 = x
            while x2 + 1 < w and int(row[x2 + 1]) == col:
                x2 += 1
            if col != bg and x2 - x + 1 >= MIN_FRAME and x2 - x + 1 < w:
                runs.setdefault(col, []).append((y, x, x2))
            x = x2 + 1
    boxes = []
    for col, rs in runs.items():
        tops = {(y, a) for y, a, _ in rs}
        for y1, a1, b1 in rs:
            if (y1 - 1, a1) in tops:
                continue  # not the top of a thick bar
            for y2 in range(y1 + 6, min(y1 + 17, h)):
                # the lower side mostly of the colour (something may cross it), and
                # sides of the colour joining the two
                if (grid[y2, a1:b1 + 1] == col).mean() < 0.7:
                    continue
                left = next((x for x in (a1 - 1, a1) if x >= 0 and (grid[y1 + 1:y2, x] == col).mean() >= 0.7), None)
                right = next((x for x in (b1 + 1, b1) if x < w and (grid[y1 + 1:y2, x] == col).mean() >= 0.7), None)
                if left is not None and right is not None:
                    boxes.append((col, (left, y1, right, y2)))
                break
    if not boxes:
        return None
    squares = [(c, _square(c)) for c in comps]
    squares = [(c, k) for c, k in squares if k is not None]

    def inside(c: Comp, box) -> bool:
        x1, y1, x2, y2 = box
        return x1 < c.x and c.x + c.w - 1 < x2 and y1 < c.y and c.y + c.h - 1 < y2

    frames = []
    for col, box in boxes:
        cells = _cells(grid, bg, box)
        if cells:
            frames.append(Frame(col, box, cells))
    if not frames:
        return None
    frames.sort(key=lambda f: (f.box[1], f.box[0]))
    in_frame = lambda c: any(inside(c, f.box) for f in frames)  # noqa: E731
    sizes = Counter(c.w for c, k in squares if k == "hollow" and not in_frame(c) and c.w >= 5)
    if not sizes:
        return None
    size = sizes.most_common(1)[0][0]
    targets = sorted((c for c, k in squares if k == "hollow" and c.w == size and not in_frame(c)),
                     key=lambda c: (c.y, c.x))
    top = min(f.box[1] for f in frames)
    targets = [c for c in targets if c.y + c.h <= top]
    tile_sizes = {c.w for f in frames for c, k in squares if inside(c, f.box) and 3 < c.w <= 6}
    palette = [Cell("tile" if k == "solid" else "call", c.color, _centre(c)) for c, k in squares
               if not in_frame(c) and c.y > top and 3 < c.w <= 6 and (not tile_sizes or c.w in tile_sizes)]
    if len(targets) < 2:
        return None
    return [c.color for c in targets], frames, palette


def run_program(frames: list[Frame], content: dict, targets: list[int]) -> int:
    """How many targets a run writes before it stops (len(targets) when it completes)."""
    by_color = {f.color: i for i, f in enumerate(frames)}
    stack = [[0, 0]]
    t = 0
    seen = set()
    for _ in range(MAX_STEPS):
        if t == len(targets):
            return t
        fi, si = stack[-1]
        if si >= len(frames[fi].slots):
            if len(stack) == 1:
                return t
            stack.pop()
            stack[-1][1] += 1
            continue
        cell = content.get((fi, si), frames[fi].slots[si])
        if cell.kind == "empty":
            return t
        if cell.kind == "tile":
            if cell.color != targets[t]:
                return t
            t += 1
            stack[-1][1] += 1
            continue
        if cell.color not in by_color:
            return t
        key = (tuple(map(tuple, stack)), t)
        if key in seen:
            return t  # going round without writing anything
        seen.add(key)
        stack.append([by_color[cell.color], 0])
    return t


def plan_program(frames: list[Frame], palette: list[Cell], targets: list[int]) -> list[tuple[Cell, tuple[int, int]]] | None:
    """The fewest placements (palette cell, slot) after which a run writes the target."""
    by_color = {f.color: i for i, f in enumerate(frames)}
    best: list = [None]
    nodes = [0]

    def search(stack, t, content, left, placed, seen):
        nodes[0] += 1
        if nodes[0] > MAX_NODES:
            return
        if best[0] is not None and len(placed) >= len(best[0]):
            return
        stack = [list(s) for s in stack]
        seen = set(seen)
        for _ in range(MAX_STEPS):
            if t == len(targets):
                best[0] = list(placed)
                return
            fi, si = stack[-1]
            if si >= len(frames[fi].slots):
                if len(stack) == 1:
                    return
                stack.pop()
                stack[-1][1] += 1
                continue
            cell = content.get((fi, si), frames[fi].slots[si])
            if cell.kind == "empty":
                # a choice: the colour wanted next, or a call
                options = []
                for i, p in enumerate(left):
                    if p is None:
                        continue
                    if (p.kind == "tile" and p.color == targets[t]) or (p.kind == "call" and p.color in by_color):
                        if all(q is None or (q.kind, q.color) != (p.kind, p.color) for q in left[:i]):
                            options.append(i)  # one of each kind and colour
                for i in options:
                    p = left[i]
                    new_left = list(left)
                    new_left[i] = None
                    new_content = dict(content)
                    new_content[(fi, si)] = p
                    search(stack, t, new_content, new_left, placed + [(p, cell.at)], seen)
                return
            if cell.kind == "tile":
                if cell.color != targets[t]:
                    return
                t += 1
                stack[-1][1] += 1
                continue
            if cell.color not in by_color:
                return
            key = (tuple(map(tuple, stack)), t)
            if key in seen:
                return
            seen.add(key)
            stack.append([by_color[cell.color], 0])

    search([[0, 0]], 0, {}, list(palette), [], set())
    return best[0]


class ProgramSolver:
    name = "program"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.fails = 0
        self.queue: list[Action] = []
        self.expect: tuple | None = None  # (palette cell, slot) being placed
        self.ran = False
        self.read = False  # the level's first frame has been looked at

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
            self.queue, self.expect = [], None
            return
        if self.in_control and action is RUN and self.ran:
            # the run did not complete the level: the reading of the program is wrong here
            self.stats["failed_runs"] += 1
            self._give_up()
            return
        if self.in_control and self.expect is not None and not self.queue:
            # a placement just ended: the slot must now hold the tile
            tile, slot = self.expect
            self.expect = None
            grid = np.asarray(obs.frame[-1])
            if not self._holds(grid, slot, tile):
                self.fails += 1
                self.stats["mismatches"] += 1

    @staticmethod
    def _holds(grid: np.ndarray, at: tuple[int, int], tile: Cell) -> bool:
        bg = int(np.bincount(grid.ravel()).argmax())
        c = _cell_at(grid, bg, at[0], at[1])
        return c is not None and (c.kind, c.color) == (tile.kind, tile.color)

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        avail = set(obs.available_actions)
        if 6 not in avail or RUN.value not in avail:
            return None
        if self.fails >= MAX_FAILS:
            self._give_up()
            return None
        if not self.queue:
            grid = np.asarray(obs.frame[-1])
            mask = fp.status_bar_mask(fp.segment(grid))
            screen = read_screen(grid, mask)
            if screen is None:
                # not a program (looked at once a level), or no longer readable
                self.gave_up = not self.read or self.in_control
                self.in_control = False
                self.read = True
                return None
            self.read = True
            targets, frames, palette = screen
            content: dict = {}
            if run_program(frames, content, targets) == len(targets):
                self.ran = True
                self.stats["runs"] += 1
                self.in_control = True
                return RUN, None
            plan = plan_program(frames, palette, targets)
            if not plan:
                self.stats["no_plan"] += 1
                if self.in_control:
                    self._give_up()
                return None
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
            tile, slot = plan[0]
            self.expect = (tile, slot)
            self.queue = [(GameAction.ACTION6, {"x": tile.at[0], "y": tile.at[1]}),
                          (GameAction.ACTION6, {"x": slot[0], "y": slot[1]})]
            self.stats["placements"] += 1
        self.stats["clicks"] += 1
        return self.queue.pop(0)
