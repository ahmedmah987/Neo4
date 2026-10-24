"""The connector family: put the pieces together so that every connector meets another.

Some puzzles show pieces on a grid, each with connector cells (one colour, the same in
every piece) sticking out of its body. A click selects a piece; four keys move the
selected piece a cell; a fifth turns it a quarter, or, for a piece that is a stack of
variants, shows the next variant. The level is done when every connector lies on exactly
one connector of another piece (of its own type; the types are not shown).

From the frames alone:

* the grid: the largest cell size (and offset) at which every drawn cell is of one colour;
* the pieces: groups of drawn cells; connectors are the cells of the colour every piece
  shows (learned on the first level that shows them all); a piece drawn all in one colour
  (unselected pieces greyed out) shows its connectors once selected, so it is selected
  once to be seen;
* what the keys do: the moves and the turn are expected as usual and checked on use; a
  turn that does not give the piece turned shows a stack, whose variants are then cycled
  through once.

The plan: placements where every connector lies on exactly one other (the types unseen)
and no body overlaps anything, found by matching the open connectors one at a time; each
group of joined pieces then goes where it costs the fewest actions (a click to select a
piece, turns, moves). A plan played out without completing the level is ruled out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

Action = tuple[GameAction, dict[str, int] | None]

MOVES = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
TURN = GameAction.ACTION5
MAX_PIECES = 8
MAX_CONFIGS = 400
MAX_NODES = 200000
MAX_FAILS = 6
MAX_LEVEL_ACTIONS = 250


# --- reading the screen ------------------------------------------------------------


@dataclass
class Lattice:
    c: int
    ox: int
    oy: int
    rows: int
    cols: int
    bg: int

    def cells(self, grid: np.ndarray) -> np.ndarray:
        c = self.c
        sub = grid[self.oy:self.oy + self.rows * c, self.ox:self.ox + self.cols * c]
        return sub.reshape(self.rows, c, self.cols, c)[:, 0, :, 0]

    def point(self, cell: tuple[int, int]) -> tuple[int, int]:
        return (self.ox + cell[0] * self.c + self.c // 2, self.oy + cell[1] * self.c + self.c // 2)


def find_lattice(grid: np.ndarray) -> Lattice | None:
    """The largest cell size and offset at which every drawn cell is of one colour."""
    H, W = grid.shape
    bg = int(np.bincount(grid.ravel()).argmax())
    for c in (6, 5, 4, 3, 2):
        for oy in range(c):
            for ox in range(c):
                rows, cols = (H - oy) // c, (W - ox) // c
                if (H - 2 * oy) % c == 0 and (H - 2 * oy) // c < rows:
                    rows = (H - 2 * oy) // c
                if (W - 2 * ox) % c == 0 and (W - 2 * ox) // c < cols:
                    cols = (W - 2 * ox) // c
                sub = grid[oy:oy + rows * c, ox:ox + cols * c].reshape(rows, c, cols, c)
                first = sub[:, :1, :, :1]
                drawn = (sub != bg).any(axis=(1, 3))
                if drawn.sum() < 4:
                    continue
                if ((sub == first).all(axis=(1, 3)) | ~drawn).all():
                    return Lattice(c, ox, oy, rows, cols, bg)
    return None


def groups(cells: np.ndarray, bg: int) -> list[set]:
    """4-connected groups of drawn cells, as sets of (x, y)."""
    rows, cols = cells.shape
    seen = np.zeros(cells.shape, dtype=bool)
    out = []
    for y in range(rows):
        for x in range(cols):
            if seen[y, x] or cells[y, x] == bg:
                continue
            stack, g = [(x, y)], set()
            seen[y, x] = True
            while stack:
                cx, cy = stack.pop()
                g.add((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < cols and 0 <= ny < rows and not seen[ny, nx] and cells[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            out.append(g)
    return out


@dataclass(frozen=True)
class Pose:
    body: frozenset  # (dx, dy) from the top-left
    conn: frozenset
    w: int
    h: int

    @classmethod
    def of(cls, body: set, conn: set) -> "Pose":
        cells = body | conn
        x0, y0 = min(x for x, _ in cells), min(y for _, y in cells)
        b = frozenset((x - x0, y - y0) for x, y in body)
        k = frozenset((x - x0, y - y0) for x, y in conn)
        return cls(b, k, max(x for x, _ in cells) - x0 + 1, max(y for _, y in cells) - y0 + 1)

    def turned(self, cw: bool) -> "Pose":
        h = self.h
        w = self.w
        if cw:
            f = lambda p: (h - 1 - p[1], p[0])  # noqa: E731
        else:
            f = lambda p: (p[1], w - 1 - p[0])  # noqa: E731
        return Pose(frozenset(map(f, self.body)), frozenset(map(f, self.conn)), h, w)

    def cells(self) -> frozenset:
        return self.body | self.conn


@dataclass
class Piece:
    x: int
    y: int
    pose: Pose
    known: bool  # its connectors have been seen
    kind: str | None = None  # "turn" or "stack", once a turn has shown which
    variants: list | None = None  # a stack's variants, in the order they came
    vi: int = 0
    seq: list | None = None  # the variants shown press after press
    line: list | None = None  # the variants end to end, once both ends were seen
    marked: int = 0  # body colours seen while selected (a stack shows one more)

    def cells(self) -> set:
        return {(self.x + dx, self.y + dy) for dx, dy in self.pose.cells()}


# --- the planner -------------------------------------------------------------------


def _fits(placed: dict) -> tuple[bool, list]:
    """Whether no body overlaps anything and no cell holds more than two connectors;
    and the connectors still alone."""
    conn: dict = {}
    body: set = set()
    for i, (pose, ox, oy) in placed.items():
        for dx, dy in pose.body:
            c = (ox + dx, oy + dy)
            if c in body:
                return False, []
            body.add(c)
        for dx, dy in pose.conn:
            conn.setdefault((ox + dx, oy + dy), []).append(i)
    for c, owners in conn.items():
        if c in body or len(owners) > 2 or (len(owners) == 2 and owners[0] == owners[1]):
            return False, []
    alone = sorted((c, owners[0]) for c, owners in conn.items() if len(owners) == 1)
    return True, alone


def configurations(poses: list[list[Pose]], limit: int = MAX_CONFIGS) -> list[dict]:
    """Relative placements of all pieces (index -> (pose, x, y)) in which every connector
    meets exactly one other and no body overlaps anything."""
    n = len(poses)
    out: list = []
    nodes = [0]

    def dfs(placed: dict) -> None:
        if len(out) >= limit or nodes[0] > MAX_NODES:
            return
        nodes[0] += 1
        ok, alone = _fits(placed)
        if not ok:
            return
        if not alone:
            if len(placed) == n:
                out.append(dict(placed))
                return
            i = min(set(range(n)) - set(placed))
            for pose in poses[i]:
                placed[i] = (pose, 0, 0)
                dfs(placed)
                del placed[i]
            return
        (x, y), owner = alone[0]
        for j in range(n):
            if j in placed:
                continue
            for pose in poses[j]:
                for cx, cy in pose.conn:
                    placed[j] = (pose, x - cx, y - cy)
                    dfs(placed)
                    del placed[j]

    dfs({})
    return out


# --- the solver ----------------------------------------------------------------------


class ConnectSolver:
    name = "connect"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.conn_colour: int | None = None
        self.moves = dict(MOVES)
        self.cw: bool | None = None  # which way a turn goes, once seen
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.fails = 0
        self.actions = 0
        self.lat: Lattice | None = None
        self.pieces: list[Piece] = []
        self.selected: int | None = None
        self.sure = False  # the selection is known, not guessed
        self.queue: list = []
        self.expect: tuple | None = None
        self.ruled_out: list = []
        self.target: dict | None = None
        self.exploring: int | None = None  # a stack being cycled through

    # --- reading -------------------------------------------------------------------

    def _read(self, grid: np.ndarray) -> bool:
        lat = find_lattice(grid)
        if lat is None:
            return False
        cells = lat.cells(grid)
        gs = groups(cells, lat.bg)
        if not 2 <= len(gs) <= MAX_PIECES or any(len(g) < 2 for g in gs):
            return False
        colours = [Counter(int(cells[y, x]) for x, y in g) for g in gs]
        if self.conn_colour is None:
            common = set(colours[0])
            for c in colours[1:]:
                common &= set(c)
            if len(common) != 1:
                return False
            self.conn_colour = common.pop()
        k = self.conn_colour
        if not any(k in c for c in colours):
            return False
        pieces = []
        for g, cs in zip(gs, colours):
            conn = {p for p in g if cells[p[1], p[0]] == k}
            known = bool(conn) or len(cs) > 1
            body = g - conn
            if not body:
                return False
            pose = Pose.of(body, conn)
            pieces.append(Piece(min(x for x, _ in g), min(y for _, y in g), pose, known))
        self.lat = lat
        self.pieces = pieces
        # the selected piece: the one drawn unlike all the others, if there is one
        looks = [frozenset(c) - {k} for c in colours]
        odd = [i for i, lk in enumerate(looks) if sum(1 for o in looks if o == lk) == 1]
        if len(pieces) > 2 and len(odd) == 1:
            self.selected, self.sure = odd[0], False
        return True

    def _drawn(self, grid: np.ndarray) -> set:
        cells = self.lat.cells(grid)
        ys, xs = np.nonzero(cells != self.lat.bg)
        return set(zip(xs.tolist(), ys.tolist()))

    def _model(self) -> set:
        out: set = set()
        for p in self.pieces:
            out |= p.cells()
        return out

    # --- planning ------------------------------------------------------------------

    def _poses(self, p: Piece) -> list:
        """(pose, turns needed) for each pose the piece can take."""
        if p.kind == "stack" and p.variants:
            line = p.line or list(range(len(p.variants)))
            here = line.index(p.vi) if p.vi in line else 0
            return [(v, abs(line.index(k) - here) if k in line else len(line)) for k, v in enumerate(p.variants)]
        cw = True if self.cw is None else self.cw
        out, pose, seen = [], p.pose, set()
        for n in range(4):
            if pose not in seen:
                seen.add(pose)
                out.append((pose, n))
            pose = pose.turned(cw)
        return out

    def _plan(self) -> dict | None:
        """The cheapest placement: piece -> (pose, turns, x, y)."""
        options = [self._poses(p) for p in self.pieces]
        poses = [[pose for pose, _ in o] for o in options]
        turns = [{pose: t for pose, t in o} for o in options]
        best = None
        for cfg in configurations(poses):
            key = frozenset((i, pose) for i, (pose, _, _) in cfg.items())
            # components: pieces joined by connectors
            comp = {i: i for i in cfg}

            def find(a):
                while comp[a] != a:
                    a = comp[a]
                return a

            where: dict = {}
            for i, (pose, ox, oy) in cfg.items():
                for dx, dy in pose.conn:
                    where.setdefault((ox + dx, oy + dy), []).append(i)
            for owners in where.values():
                if len(owners) == 2:
                    comp[find(owners[0])] = find(owners[1])
            members: dict = {}
            for i in cfg:
                members.setdefault(find(i), []).append(i)
            total, plan, used = 0, {}, set()
            for ms in members.values():
                choice = None
                for tx in range(-self.lat.cols, self.lat.cols + 1):
                    for ty in range(-self.lat.rows, self.lat.rows + 1):
                        cost, tmp, cells, ok = 0, {}, set(), True
                        for i in ms:
                            pose, ox, oy = cfg[i]
                            X, Y = ox + tx, oy + ty
                            if X < 0 or Y < 0 or X + pose.w > self.lat.cols or Y + pose.h > self.lat.rows:
                                ok = False
                                break
                            p = self.pieces[i]
                            t = turns[i][pose]
                            mv = abs(X - p.x) + abs(Y - p.y)
                            if t or mv or pose != p.pose:
                                cost += t + mv + (0 if i == self.selected else 1)
                            tmp[i] = (pose, t, X, Y)
                            cells |= {(X + dx, Y + dy) for dx, dy in pose.cells()}
                        if not ok or cells & used:
                            continue
                        if choice is None or cost < choice[0]:
                            choice = (cost, tmp, cells)
                if choice is None:
                    total = None
                    break
                total += choice[0]
                plan.update(choice[1])
                used |= choice[2]
            if total is None:
                continue
            if any(set(plan.items()) == set(r.items()) for r in self.ruled_out):
                continue
            if best is None or total < best[0]:
                best = (total, plan)
        return None if best is None else best[1]

    def _steps(self, plan: dict) -> list:
        """The actions that carry the plan out: (kind, piece, action, data)."""
        order = sorted(plan, key=lambda i: (i != self.selected, i))
        out = []
        for i in order:
            pose, t, X, Y = plan[i]
            p = self.pieces[i]
            if not t and X == p.x and Y == p.y:
                continue
            if i != self.selected:
                out.append(("select", i))
            if p.kind == "stack":
                if pose != p.pose:
                    out.append(("variant", i, pose))
            else:
                out.extend(("turn", i) for _ in range(t))
            out.append(("goto", i, X, Y))
        return out

    # --- acting --------------------------------------------------------------------

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    def _click_on(self, i: int) -> Action | None:
        """A click on a cell of piece i that nothing else covers."""
        others: set = set()
        for j, q in enumerate(self.pieces):
            if j != i:
                others |= q.cells()
        p = self.pieces[i]
        for dx, dy in sorted(p.pose.body):
            c = (p.x + dx, p.y + dy)
            if c not in others:
                x, y = self.lat.point(c)
                return GameAction.ACTION6, {"x": x, "y": y}
        return None

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
            if action is GameAction.RESET and self.read:
                self.read, self.lat = False, None
            return
        if self.expect is None or self.lat is None:
            return
        kind, i, before = self.expect
        self.expect = None
        grid = np.asarray(obs.frame[-1])
        drawn = self._drawn(grid)
        cells = self.lat.cells(grid)
        p = self.pieces[i]
        others: set = set()
        for j, q in enumerate(self.pieces):
            if j != i:
                others |= q.cells()
        mine = drawn - others
        if kind == "select":
            # a click on a piece not selected selects it
            self.selected, self.sure = i, True
            p.marked = max(p.marked, len({int(cells[c[1], c[0]]) for c in p.cells()} - {self.conn_colour}))
            if not p.known:
                # selected, it shows itself: its connectors, if it has any
                conn = {c for c in p.cells() if cells[c[1], c[0]] == self.conn_colour}
                p.pose = Pose.of(p.cells() - conn, conn)
                p.known = True
                self.stats["revealed"] += 1
            return
        if kind == "move":
            dx, dy = before
            if drawn == others | {(c[0] + dx, c[1] + dy) for c in p.cells()}:
                p.x, p.y = p.x + dx, p.y + dy
                self.sure = True
                return
            if drawn == others | p.cells():
                # nothing moved: the piece was not selected (a click let it go)
                self.sure = False
                self.selected = None
                self.fails += 1
                self.stats["not_selected"] += 1
                self.queue = []
                return
            # it moved some other way: learn the key
            for ddx in range(-2, 3):
                for ddy in range(-2, 3):
                    if (ddx or ddy) and drawn == others | {(c[0] + ddx, c[1] + ddy) for c in p.cells()}:
                        self.moves[action] = (ddx, ddy)
                        p.x, p.y = p.x + ddx, p.y + ddy
                        self.sure = True
                        self.stats["keys_learned"] += 1
                        self.queue = []
                        return
            self.fails += 1
            self.stats["move_missed"] += 1
            self.queue = []
            return
        if kind == "turn":
            if drawn == others | p.cells() and p.kind != "stack":
                self.sure = False
                self.selected = None
                self.fails += 1
                self.stats["not_selected"] += 1
                self.queue = []
                return
            self.sure = True

            def placed(pose: Pose) -> tuple:
                # turned or swapped in place, then pushed back inside the grid
                x = max(0, min(p.x, self.lat.cols - pose.w))
                y = max(0, min(p.y, self.lat.rows - pose.h))
                return x, y, {(x + dx, y + dy) for dx, dy in pose.cells()}

            if p.kind != "stack":
                for cw in ((self.cw,) if self.cw is not None else (True, False)):
                    turned = p.pose.turned(cw)
                    x, y, mine_t = placed(turned)
                    if drawn == others | mine_t:
                        self.cw = cw
                        p.kind = "turn"
                        p.pose, p.x, p.y = turned, x, y
                        return
            else:
                for k, v in enumerate(p.variants):
                    x, y, mine_v = placed(v)
                    if drawn == others | mine_v and k != p.vi:
                        self._shown(i, k, x, y)
                        return
            if not mine:
                self.fails += 1
                self.queue = []
                return
            conn = {c for c in mine if cells[c[1], c[0]] == self.conn_colour}
            new = Pose.of(mine - conn, conn)
            nx, ny = min(x for x, _ in mine), min(y for _, y in mine)
            if p.kind != "stack":
                # not the piece turned: a stack of variants
                p.kind = "stack"
                p.variants = [p.pose, new]
                p.vi, p.seq = 1, [0, 1]
                p.pose, p.x, p.y = new, nx, ny
                self.exploring = i
                self.queue = []
                self.stats["stacks"] += 1
                return
            seen = next((k for k, v in enumerate(p.variants) if v.cells() == new.cells()), None)
            if seen is None:
                p.variants.append(new)
                seen = len(p.variants) - 1
            self._shown(i, seen, nx, ny)

    def _shown(self, i: int, k: int, x: int, y: int) -> None:
        """A stack now shows its variant k at (x, y)."""
        p = self.pieces[i]
        p.vi = k
        p.pose, p.x, p.y = p.variants[k], x, y
        p.seq = (p.seq or []) + [k]
        # a variant shown again after one other: that other is an end of the line
        ends = [j for j in range(2, len(p.seq)) if p.seq[j] == p.seq[j - 2]]
        if self.exploring == i and len(ends) >= 2:
            a, b = ends[0] - 1, ends[1] - 1
            p.line = p.seq[a:b + 1]
            self.exploring = None

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        avail = set(obs.available_actions)
        if not {1, 2, 3, 4, 5, 6} <= avail:
            return None
        if self.fails >= MAX_FAILS or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if self.lat is None:
            if self.read:
                return None
            self.read = True
            if not self._read(grid):
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
        if not self.queue:
            # see every piece's connectors first
            hidden = next((i for i, p in enumerate(self.pieces) if not p.known), None)
            if self.exploring is not None:
                q = self.pieces[self.exploring]
                if len(q.seq or []) > 4 * len(q.variants) + 4:
                    self.exploring = None  # no second end: take the line as seen
                    q.line = list(dict.fromkeys(q.seq))
                    return self.act(obs)
                self.queue = [("select", self.exploring)] if self.selected != self.exploring else []
                self.queue.append(("turn", self.exploring))
            elif hidden is not None:
                self.queue = [("select", hidden)]
            else:
                plan = self._plan()
                if plan is None:
                    # a piece whose turn has not been tried may be a stack
                    untried = min((i for i, p in enumerate(self.pieces) if p.kind is None),
                                  key=lambda i: (-self.pieces[i].marked, i), default=None)
                    if untried is None:
                        self.stats["no_plan"] += 1
                        self._give_up()
                        return None
                    self.queue = ([] if untried == self.selected else [("select", untried)]) + [("turn", untried)]
                else:
                    steps = self._steps(plan)
                    if not steps:
                        # played out and the level goes on: the types do not match there
                        self.ruled_out.append(plan)
                        self.stats["ruled_out"] += 1
                        self.fails += 1
                        return self.act(obs) if self.fails < MAX_FAILS else None
                    self.queue = steps
                    self.target = plan
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        step = self.queue[0]
        if step[0] == "select":
            self.queue.pop(0)
            if self.selected == step[1]:
                return self.act(obs)  # a click on the selected piece would let it go
            a = self._click_on(step[1])
            if a is None:
                self._give_up()
                return None
            self.expect = ("select", step[1], None)
            return self._do(a)
        if step[0] == "turn":
            self.queue.pop(0)
            self.expect = ("turn", step[1], None)
            return self._do((TURN, None))
        if step[0] == "variant":
            _, i, pose = step
            p = self.pieces[i]
            if p.pose.cells() == pose.cells():
                self.queue.pop(0)
                return self.act(obs)
            if len(p.seq or []) > 8 * len(p.variants) + 8:
                self._give_up()
                return None
            self.expect = ("turn", i, None)
            return self._do((TURN, None))
        # goto: one move towards the place
        _, i, X, Y = step
        p = self.pieces[i]
        if (p.x, p.y) == (X, Y):
            self.queue.pop(0)
            return self.act(obs) if self.queue else self._replan(obs)
        want = (1 if X > p.x else -1 if X < p.x else 0, 0) if X != p.x else (0, 1 if Y > p.y else -1)
        key = next((k for k, d in self.moves.items() if d == want), None)
        if key is None:
            self._give_up()
            return None
        self.expect = ("move", i, self.moves[key])
        return self._do((key, None))

    def _replan(self, obs: FrameDataRaw) -> Action | None:
        return self.act(obs)

    def _do(self, a: Action) -> Action:
        self.actions += 1
        self.stats["actions"] += 1
        return a
