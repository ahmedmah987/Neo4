"""The replay family: walk the avatar to the exit through doors that plates hold open while
something stands on them; the fifth key sends the world back to the start and leaves a ghost
that walks the run again, a move for each of the avatar's.

Some puzzles show a maze of corridors, an avatar, an exit boxed in the avatar's colour, and
devices wired together in a colour of their own: plates (a small square in the middle of a
cell), doors (a cell filled with the colour) and pads (a cell framed in it). A door is open
while something stands on a plate wired to it; some doors turn instead, open or shut at each
step onto their plate; a plate wired to two pads swaps what stands on them. The fifth key
undoes the run: the world goes back to how the level began, and what the avatar did becomes
a ghost that does it again, one move for each move of the avatar (a move the avatar cannot
make does not count). A row of icons outside the maze counts the runs; past the last, the
ghosts are gone. Enemies, squares of the avatar's shape in another colour, step once for each
move of the avatar: on along their way, else the turn to their right, to their left, or back,
and back from a shut door. They press plates too, and catch the avatar by stepping onto it.

From the frames alone:

* the avatar is the one square of one colour (its look is kept for the game); its cells are a
  step apart, the step its size and a pixel; the exit is the cell boxed in its colour;
* a cell is open when the avatar's box there shows no background; doors, plates and pads are
  read by their shapes, and what one region of a colour joins acts together;
* a door that stays open with nothing on its plates turns; one that shuts holds;
* an enemy walks the open cells, facing down until its first step shows its way; a cell it
  is seen not to step onto is off its way; it keeps its heading when the run is undone;
* the ghosts and the runs left are taken as the fifth key gives them, and checked when seen.

The plan: for each choice of cells for the ghosts to end on (plates and pads, as many as the
runs allow), a breadth-first search over the avatar's cell, the ghosts' cells, the doors, the
enemies and the time, for the shortest run onto each in turn and then onto the exit; the
cheapest plan in actions is played. A run that will be undone is one the undo brings back
cleanly (see World). A plan may begin by restarting the level when what has been done so far
is no use or not known. With no plan, a plate of a colour whose doors have not been seen with
the plate left is stepped on and off, a colour whose turning would make a plan first.
"""

from __future__ import annotations

from collections import Counter, deque
from itertools import product

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
REWIND = GameAction.ACTION5
MAX_FAILS = 6
MAX_LEVEL_ACTIONS = 300
MAX_STATES = 200_000
MAX_GHOSTS = 3  # ghosts planned when the runs cannot be read
SQUARE = (4, 8)  # the sizes an avatar is looked for in


# --- reading the screen ------------------------------------------------------------


def find_avatar(grid: np.ndarray, look=None):
    """(x, y, size, colour) of the avatar: the one square of one colour (a pixel or two of
    another inside it aside), or the square of the look kept."""
    bg = Counter(grid.ravel().tolist()).most_common(1)[0][0]
    found = []
    for c in fp.segment(grid):
        if c.color == bg or c.width != c.height or not SQUARE[0] <= c.width <= SQUARE[1]:
            continue
        if c.area < 0.9 * c.width * c.width:
            continue
        if look is not None and (c.color, c.width) != look:
            continue
        found.append((c.x1, c.y1, c.width, c.color))
    if look is None:
        colours = Counter(f[3] for f in found)
        found = [f for f in found if colours[f[3]] == 1]
    return found[0] if len(found) == 1 else None


def read_cells(grid: np.ndarray, x0: int, y0: int, s: int, step: int, bg: int, floor: int, mask=None):
    """Every cell of the lattice: (x, y) -> (kind, colour), kind one of open, door, plate, pad
    (a wall is left out). `mask` marks pixels of movers, read as floor."""
    H, W = grid.shape
    g = grid.astype(np.int64)
    if mask is not None:
        g = np.where(mask, floor, g)
    out = {}
    for y in range(y0 % step, H - s + 1, step):
        for x in range(x0 % step, W - s + 1, step):
            inner = g[y:y + s, x:x + s]
            if (inner == bg).any():
                continue
            vals, counts = np.unique(inner, return_counts=True)
            top = sorted(((n, int(v)) for v, n in zip(vals, counts) if v != floor), reverse=True)
            if top and top[0][0] >= 0.75 * s * s:
                out[(x, y)] = ("door", top[0][1])
                continue
            core = inner[1:s - 1, 1:s - 1]
            ring = np.concatenate([inner[0], inner[-1], inner[1:-1, 0], inner[1:-1, -1]])
            c0 = int(core.flat[0])
            if len(np.unique(core)) == 1 and c0 != floor and np.isin(ring, (floor, c0)).all() and (ring == floor).mean() >= 0.5:
                out[(x, y)] = ("plate", c0)  # a wire may leave it through its ring
                continue
            if x >= 1 and y >= 1 and x + s < W and y + s < H:
                box = g[y - 1:y + s + 1, x - 1:x + s + 1]
                edge = np.concatenate([box[0], box[-1], box[1:-1, 0], box[1:-1, -1]])
                if len(np.unique(edge)) == 1 and edge[0] not in (floor, bg) and (inner == floor).all():
                    out[(x, y)] = ("pad", int(edge[0]))
                    continue
            out[(x, y)] = ("open", None)
    return out


def find_goal(grid: np.ndarray, cells: dict, s: int, colour: int):
    """The open cell whose box is drawn round in the avatar's colour (two sides or more)."""
    H, W = grid.shape
    found = []
    for (x, y), (kind, _) in cells.items():
        if kind != "open" or x < 1 or y < 1 or x + s >= W or y + s >= H:
            continue
        box = grid[y - 1:y + s + 1, x - 1:x + s + 1]
        edge = np.concatenate([box[0], box[-1], box[1:-1, 0], box[1:-1, -1]])
        if (edge == colour).sum() >= 2 * (s + 2):
            found.append((x, y))
    return found[0] if len(found) == 1 else None


def read_groups(grid: np.ndarray, cells: dict, s: int) -> list:
    """What is wired together: per region of a device's colour, its plates, doors and pads."""
    groups = {}
    for colour in {c for k, c in cells.values() if k in ("door", "plate", "pad")}:
        lab = {}
        for comp in fp.segment(np.where(grid == colour, 1, 0)):
            if comp.color != 1:
                continue
            for y, x in comp.pixels.tolist():
                lab[(x, y)] = comp.id
        for (x, y), (kind, c) in cells.items():
            if c != colour:
                continue
            probe = (x + s // 2, y + s // 2) if kind in ("door", "plate") else (x - 1, y - 1)
            cid = lab.get(probe)
            if cid is None:
                continue
            g = groups.setdefault((colour, cid), {"colour": colour, "plates": [], "doors": [], "pads": []})
            g[kind + "s"].append((x, y))
    return [g for g in groups.values() if g["plates"] and (g["doors"] or len(g["pads"]) == 2)]


def read_runs(grid: np.ndarray, cells: dict, s: int):
    """The runs a row of small square icons outside the maze counts, or None."""
    icons = [c for c in fp.segment(grid) if c.width == 3 and c.height == 3 and c.area >= 8
             and not any(x <= c.x1 and c.x2 < x + s and y <= c.y1 and c.y2 < y + s for x, y in cells)]
    rows = Counter(c.y1 for c in icons)
    if not rows:
        return None
    y, n = rows.most_common(1)[0]
    return n if n >= 2 else None


def find_shapes(grid: np.ndarray, shape: np.ndarray, colours) -> list:
    """(x, y, colour) of every region drawn in the avatar's shape in one of the colours."""
    s = shape.shape[0]
    out = []
    for c in fp.segment(grid):
        if c.color not in colours or c.width != s or c.height != s or c.area != int(shape.sum()):
            continue
        if (grid[c.y1:c.y1 + s, c.x1:c.x1 + s] == c.color).tolist() == shape.tolist():
            out.append((c.x1, c.y1, c.color))
    return out


# --- the model ---------------------------------------------------------------------


def step_to(cell, d, step: int):
    return cell[0] + d[0] * step, cell[1] + d[1] * step


DOWN = (0, 1)


class World:
    """The level's devices and how a run plays out, a move at a time.

    A state is (avatar, ghosts, pressed, turned, foes): the cells of the avatar and of each
    ghost, which groups have something on a plate, which turning groups are turned, and each
    enemy's (cell, heading, alive).

    A run the fifth key will undo is kept to what undoing brings back to the start: nothing
    held back by a door or stopped, nothing a door shuts on, no step that could not be walked
    back, no swap of something walked back before what pressed its plate (the undo walks the
    enemies back first, then the ghosts, then the avatar, and swaps back as the plate is
    left), and nothing left standing on a turning plate at its end (a turning door turns at
    each step onto its plate, either way, so a run that ends on one leaves it turned)."""

    def __init__(self, cells: dict, groups: list, step: int, kinds: dict, start, foes=(), off_track=frozenset()) -> None:
        self.step = step
        self.start = start
        self.open = {c for c, (k, _) in cells.items() if k != "door"}
        self.groups = groups
        self.door_of: dict = {}
        self.plate_of: dict = {}
        self.pads: list = []
        for gi, g in enumerate(groups):
            for c in g["doors"]:
                if c in cells:
                    self.door_of[c] = gi
                    self.open.add(c)
            for c in g["plates"]:
                self.plate_of[c] = gi
            if len(g["pads"]) == 2:
                self.pads.append((gi, tuple(g["pads"])))
        # a group holds its doors open while pressed, or turns them at each press (by colour)
        self.turns = [kinds.get(g["colour"]) == "turn" for g in groups]
        self.foes = tuple(foes)  # where each enemy begins
        self.track = self.open - set(off_track)  # where enemies walk

    def shut(self, c, pressed: tuple, turned: tuple) -> bool:
        gi = self.door_of.get(c)
        if gi is None:
            return False
        return not (turned[gi] if self.turns[gi] else pressed[gi])

    def free(self, c, pressed: tuple, turned: tuple) -> bool:
        return c in self.open and not self.shut(c, pressed, turned)

    def begin(self, ghosts: int, heads: tuple) -> tuple:
        n = len(self.groups)
        return self.start, (self.start,) * ghosts, (0,) * n, (0,) * n, tuple((c, h, True) for c, h in zip(self.foes, heads))

    def foe_step(self, c, h, pressed: tuple, turned: tuple):
        """An enemy's step: on along its way, else the turn to its right, to its left, or back,
        and back from a shut door. (cell, heading, moved)."""
        for _ in range(4):
            for nh in (h, (-h[1], h[0]), (h[1], -h[0]), (-h[0], -h[1])):
                if step_to(c, nh, self.step) in self.track:
                    h = nh
                    break
            else:
                return c, h, False
            nc = step_to(c, h, self.step)
            if self.free(nc, pressed, turned):
                return nc, h, True
            h = (-h[0], -h[1])
        return c, h, False

    def advance(self, state: tuple, t: int, d, records: list, strict: bool = False):
        """One move of the avatar, the t-th move of each ghost that has one and a step of each
        enemy: the state after it, or None when the avatar cannot make it, is caught or has a
        door shut on it, or (strict) the run would not undo cleanly."""
        avatar, ghosts, pressed, turned, foes = state
        to = step_to(avatar, d, self.step)
        if not self.free(to, pressed, turned):
            return None
        old = [avatar]
        movers = [to]
        for gh, rec in zip(ghosts, records):
            old.append(gh)
            if t < len(rec):
                gt = step_to(gh, rec[t], self.step)
                if self.free(gt, pressed, turned):
                    gh = gt
                elif strict:
                    return None
            movers.append(gh)
        nfoes = []
        for c, h, alive in foes:
            if alive:
                nc, h, moved = self.foe_step(c, h, pressed, turned)
                if moved and nc == to:
                    return None  # caught
                if strict and not moved:
                    return None
                old.append(c)
                c = nc
            nfoes.append((c, h, alive))
        live = [i for i, f in enumerate(nfoes) if f[2]]
        cells = movers + [nfoes[i][0] for i in live]
        now = [0] * len(self.groups)
        for m in cells:
            gi = self.plate_of.get(m)
            if gi is not None:
                now[gi] = 1
        turned = list(turned)
        for gi in range(len(self.groups)):
            if now[gi] and not pressed[gi]:
                if self.turns[gi]:
                    turned[gi] ^= 1
                for pg, (a, b) in self.pads:
                    if pg == gi:
                        swapped = [b if m == a else a if m == b else m for m in cells]
                        if strict and swapped != cells:
                            # the undo swaps back when the plate is left, which catches only
                            # what is walked back after whatever pressed it
                            order = [len(foes) + len(ghosts)] + [len(foes) + i for i in range(len(ghosts))] + live
                            moved = [order[i] for i, m in enumerate(cells) if m in (a, b)]
                            pressers = [order[i] for i, m in enumerate(cells) if self.plate_of.get(m) == gi]
                            if min(moved) < max(pressers):
                                return None
                        cells = swapped
        now, turned = tuple(now), tuple(turned)
        if self.shut(cells[0], now, turned):
            return None  # a door shuts on the avatar
        for j, i in enumerate(live):
            c, h, _ = nfoes[i]
            c = cells[len(movers) + j]
            dead = self.shut(c, now, turned)  # a door shuts on it
            if dead and strict:
                return None
            nfoes[i] = (c, h, not dead)
        if strict and any(self.shut(c, now, turned) for c in old):
            return None  # a step that could not be walked back
        return cells[0], tuple(cells[1:len(movers)]), now, turned, tuple(nfoes)

    def settled(self, state: tuple) -> bool:
        """Nothing stands on a turning plate."""
        cells = (state[0],) + state[1] + tuple(c for c, _, alive in state[4] if alive)
        return all(not self.turns[self.plate_of[m]] for m in cells if m in self.plate_of)

    def run(self, records: list, heads: tuple, prefix: list, target: set, final: bool):
        """The fewest moves, after the prefix, from the start until the avatar is on a cell of
        target: the whole run's moves, or None. A run that is not the last is kept one the
        fifth key undoes cleanly."""
        strict = not final
        state, t = self.begin(len(records), heads), 0
        for d in prefix:
            state = self.advance(state, t, d, records, strict)
            if state is None:
                return None
            t += 1

        def done(st) -> bool:
            return st[0] in target and (final or self.settled(st))

        if t and done(state):
            return list(prefix)
        horizon = max([len(r) for r in records] + [0])
        key0 = (state, min(t, horizon))
        prev = {key0: None}
        dq = deque([(state, t)])
        while dq:
            state, t = dq.popleft()
            key = (state, min(t, horizon))
            for d in KEYS.values():
                nxt = self.advance(state, t, d, records, strict)
                if nxt is None:
                    continue
                nkey = (nxt, min(t + 1, horizon))
                if nkey in prev:
                    continue
                prev[nkey] = (key, d)
                if done(nxt):
                    moves = []
                    k = nkey
                    while prev[k] is not None:
                        k, dd = prev[k]
                        moves.append(dd)
                    return list(prefix) + moves[::-1]
                dq.append((nxt, t + 1))
            if len(prev) > MAX_STATES:
                return None
        return None

    def replay(self, records: list, heads: tuple, moves: list):
        """The states after each move of a run, or None from a move the model refuses."""
        state = self.begin(len(records), heads)
        out = []
        for t, d in enumerate(moves):
            state = self.advance(state, t, d, records)
            if state is None:
                return None
            out.append(state)
        return out

    def targets(self) -> list:
        """Where a run may end for its ghost to stand: plates that hold, and pads."""
        out = {c for c, gi in self.plate_of.items() if not self.turns[gi]}
        for _, pads in self.pads:
            out.update(pads)
        return sorted(out)


# --- the solver ------------------------------------------------------------------


class ReplaySolver:
    name = "replay"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.look: tuple | None = None  # (the avatar's colour, its size), kept for the game
        self.kinds: dict = {}  # a device colour -> "hold" or "turn", as seen
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.others = False  # someone else acted on the level before it was read
        self.scene = None
        self.queue: list = []
        self.expect = None
        self.actions = 0
        self.fails = 0
        self.walls: set = set()  # cells a move was refused onto though the model went on
        self.off_track: set = set()  # cells an enemy was seen not to walk onto
        self.on_track: set = set()  # cells an enemy was seen on
        self.explored: set = set()  # device colours stepped on and off to see their doors
        self.reset_last = False
        self.verified = False  # a move has gone as the model said on this level

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
        av = find_avatar(grid, self.look)
        if av is None:
            return False
        x, y, s, colour = av
        box = grid[max(0, y - 1):y + s + 1, max(0, x - 1):x + s + 1]
        ring = [int(v) for v in box.ravel() if v != colour]
        if not ring:
            return False
        floor = Counter(ring).most_common(1)[0][0]
        rest = Counter(v for v in grid.ravel().tolist() if v not in (floor, colour))
        if not rest:
            return False
        bg = rest.most_common(1)[0][0]  # the most common colour but the floor's
        shape = grid[y:y + s, x:x + s] == colour
        others = set(np.unique(grid).tolist()) - {colour, bg, floor}
        foes = [(fx, fy, fc) for fx, fy, fc in find_shapes(grid, shape, others) if (fx - x) % (s + 1) == 0 and (fy - y) % (s + 1) == 0]
        mask = np.zeros(grid.shape, dtype=bool)
        for mx, my in [(x, y)] + [(fx, fy) for fx, fy, _ in foes]:
            mask[my:my + s, mx:mx + s] = True
        cells = read_cells(grid, x, y, s, s + 1, bg, floor, mask)
        goal = find_goal(grid, cells, s, colour)
        groups = read_groups(grid, cells, s)
        if goal is None or not groups or (x, y) not in cells:
            return False
        self.scene = {
            "s": s, "step": s + 1, "colour": colour, "cells": cells, "goal": goal, "shape": shape,
            "groups": groups, "start": (x, y), "runs": read_runs(grid, cells, s),
            "foes": [(fx, fy) for fx, fy, _ in foes], "foe_colours": {fc for _, _, fc in foes},
        }
        self.on_track = set(self.scene["foes"])
        self.heads0 = tuple(DOWN for _ in foes)  # an enemy is taken to face down until it steps
        self.found_look = (colour, s)
        self._start_level()
        return True

    def _start_level(self) -> None:
        """The level as it begins (again): no ghosts, nothing done."""
        self.cell = self.scene["start"]
        self.records: list = []  # the moves of each run the ghosts replay
        self.heads = self.heads0  # the enemies' headings as this run began
        self.prefix: list = []  # the moves of this run so far
        self.seen: list = []  # where the enemies were seen after each move of this run
        self.lost = False  # what happened in this run is not known
        self._model()

    def _world(self, kinds: dict) -> World:
        sc = self.scene
        cells = {c: v for c, v in sc["cells"].items() if c not in self.walls}
        return World(cells, sc["groups"], sc["step"], kinds, sc["start"], sc["foes"], self.off_track)

    def _model(self) -> None:
        self.world = self._world(self.kinds)

    # --- planning -------------------------------------------------------------------

    def _plan(self, world: World | None = None):
        """The cheapest way to the exit from here, in actions: moves and rewinds, after a
        restart of the level when what has been done so far is no use (or not known)."""
        sc = self.scene
        world = world or self.world
        ghosts = (sc["runs"] - 1) if sc["runs"] else MAX_GHOSTS
        targets = world.targets()
        cache: dict = {}

        def run(records, heads, prefix, target, final):
            key = (tuple(tuple(r) for r in records), heads, tuple(prefix), target, final)
            if key not in cache:
                moves = world.run(records, heads, prefix, {target}, final)
                end = None
                if moves is not None and not final:
                    end = tuple(h for _, h, _ in world.replay(records, heads, moves)[-1][4])
                cache[key] = moves, end
            return cache[key]

        options = [] if self.lost else [(self.records, self.heads, self.prefix, [])]
        if (self.prefix or self.records or self.lost) and not self.reset_last:
            options.append(([], self.heads0, [], [GameAction.RESET]))  # begin the level again
        best = None
        for records0, heads0, prefix0, head in options:
            for k in range(max(0, ghosts - len(records0)) + 1):
                for choice in product(targets, repeat=k):
                    records, heads, prefix, actions = list(records0), heads0, list(prefix0), list(head)
                    for target in choice:
                        moves, end = run(records, heads, prefix, target, False)
                        if moves is None:
                            break
                        actions += [KEYS_OF[d] for d in moves[len(prefix):]] + [REWIND]
                        records, heads, prefix = records + [moves], end, []
                    else:
                        moves, _ = run(records, heads, prefix, sc["goal"], True)
                        if moves is not None:
                            actions += [KEYS_OF[d] for d in moves[len(prefix):]]
                            if best is None or len(actions) < len(best):
                                best = actions
        return best

    def _explore(self):
        """A plate whose doors have not been seen with it left: the way onto it from here and
        a step off it. A colour whose turning would make a plan is tried first."""
        sc = self.scene
        if self.lost:
            return None
        unknown = sorted({g["colour"] for g in sc["groups"] if g["doors"]} - set(self.kinds) - self.explored)
        useful = [c for c in unknown if self._plan(self._world({**self.kinds, c: "turn"})) is not None]
        world = self.world
        for colour in useful + [c for c in unknown if c not in useful]:
            plates = {p for g in sc["groups"] if g["colour"] == colour for p in g["plates"]}
            moves = world.run(self.records, self.heads, self.prefix, plates, True)
            if moves is None:
                continue
            states = world.replay(self.records, self.heads, moves)
            for d in KEYS.values():
                st = world.advance(states[-1], len(moves), d, self.records)
                if st is not None and st[0] not in world.door_of and st[0] not in world.plate_of:
                    self.explored.add(colour)
                    self.stats["explore"] += 1
                    return [KEYS_OF[x] for x in moves[len(self.prefix):]] + [KEYS_OF[d]]
        return None

    # --- acting ----------------------------------------------------------------------

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4, 5} <= set(obs.available_actions):
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        if self.scene is None:
            if self.read or self.others:
                return None
            self.read = True
            if not self._setup(np.asarray(obs.frame[-1])):
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
            self.stats["scenes"] += 1
        if not self.queue:
            plan = self._plan()
            if plan is None:
                plan = self._explore()
            if plan is None and self.lost and not self.reset_last:
                plan = [GameAction.RESET]
            if not plan:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            self.queue = plan
            self.stats["plans"] += 1
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        a = self.queue.pop(0)
        self.expect = (a, self.cell)
        self.actions += 1
        self.stats["actions"] += 1
        return a, None

    # --- observing --------------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty():
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
        if obs.state != GameState.NOT_FINISHED:
            self.queue = []  # the attempt is over (the time ran out, or the avatar was caught)
            return
        if action is GameAction.RESET:
            self.reset_last = True
            if exp is None or exp[0] is not GameAction.RESET:
                self.queue = []
            if self.scene is not None:
                self._start_level()
            return
        if action is None:
            return
        self.reset_last = False
        if exp is None or exp[0] is not action:
            if self.scene is None:
                self.others = True
            else:
                self.lost = True
            return
        a, cell0 = exp
        sc = self.scene
        grid = np.asarray(obs.frame[-1])
        av = find_avatar(grid, self.found_look)
        if av is None:
            self.lost = True
            self._fail("avatar_lost")
            return
        now = (av[0], av[1])
        if a is REWIND:
            if now != sc["start"]:
                self._fail("rewind_mismatch")
                self._give_up()
                return
            runs = sc["runs"] or MAX_GHOSTS + 1
            states = self.world.replay(self.records, self.heads, self.prefix)
            if states:
                self.heads = tuple(h for _, h, _ in states[-1][4])
            self.records = self.records + [self.prefix] if len(self.records) + 1 < runs else []
            self.prefix, self.seen = [], []
            self.cell = now
            self.stats["rewinds"] += 1
            return
        d = KEYS[a]
        states = self.world.replay(self.records, self.heads, self.prefix + [d])
        if now == cell0:
            if states is not None:
                # refused where the model went on: that cell is not open
                self.walls.add(step_to(cell0, d, sc["step"]))
                self._model()
                self._fail("refused")
            return
        self.prefix.append(d)
        self.seen.append(self._foes_now(grid))
        self.cell = now
        if states is not None and states[-1][0] == now and not self._foes_differ(states):
            self.verified = True
            self._learn_doors(grid, states)
            return
        if states is not None and states[-1][0] == now and self._learn_foes():
            self.verified = True
            self.queue = []  # planned again under what was learned
            return
        self.lost = True
        self._fail("mismatch")
        if not self.verified:
            self._give_up()  # the first move did not go as read: not this family

    def _foes_now(self, grid: np.ndarray):
        sc = self.scene
        if not sc["foes"]:
            return ()
        return tuple(sorted((x, y) for x, y, _ in find_shapes(grid, sc["shape"], sc["foe_colours"])))

    def _foes_differ(self, states: list) -> bool:
        """Whether the enemies seen after some move of this run are not where the model has
        them (one hidden under a ghost is not counted against it)."""
        for st, seen in zip(states, self.seen):
            model = sorted(c for c, _, alive in st[4] if alive)
            if seen != tuple(model) and not (len(seen) < len(model) and set(seen) <= set(model)):
                return True
        return False

    def _learn_foes(self) -> bool:
        """An enemy that went elsewhere than the model said: its first heading, or a cell off
        its way. True when the model then agrees with all that was seen in this run."""
        sc = self.scene
        for seen in self.seen:
            self.on_track.update(seen)
        if not self.records and len(self.prefix) == 1 and sc["foes"]:
            for heads in product([DOWN, (1, 0), (0, -1), (-1, 0)], repeat=len(sc["foes"])):
                states = self.world.replay([], heads, self.prefix)
                if states and not self._foes_differ(states):
                    self.heads0 = self.heads = heads
                    self.stats["learned_heading"] += 1
                    return True
        for _ in range(8):
            states = self.world.replay(self.records, self.heads, self.prefix)
            if states is None:
                return False
            wrong = None
            for st, seen in zip(states, self.seen):
                model = sorted(c for c, _, alive in st[4] if alive)
                extra = [c for c in model if c not in seen and c not in self.on_track]
                if seen != tuple(model) and extra:
                    wrong = extra[0]
                    break
            if wrong is None:
                return not self._foes_differ(states)
            self.off_track.add(wrong)
            self.stats["learned_off_track"] += 1
            self._model()
        return False

    def _learn_doors(self, grid: np.ndarray, states: list) -> None:
        """A door seen open after its plates were left turns; one seen shut holds."""
        sc = self.scene
        s = sc["s"]
        pressed = states[-1][2]
        for gi, g in enumerate(sc["groups"]):
            if not g["doors"] or g["colour"] in self.kinds or pressed[gi]:
                continue
            if not any(st[2][gi] for st in states[:-1]):
                continue
            shut = [(grid[y:y + s, x:x + s] == g["colour"]).sum() >= 0.75 * s * s for x, y in g["doors"]]
            if all(shut):
                self.kinds[g["colour"]] = "hold"
                self.stats["learned_hold"] += 1
            elif not any(shut):
                self.kinds[g["colour"]] = "turn"
                self.stats["learned_turn"] += 1
                self._model()
                self.queue = []


KEYS_OF = {d: k for k, d in KEYS.items()}
