"""The maze family: walk the avatar through a maze of nodes and links to the exit, past
watchers that act by rules of their own.

Some puzzles draw a maze as nodes joined by links on a lattice. The avatar is a block with
an eye on the side it faces, the exit a block of a colour of its own; other eyed blocks,
watchers, act each time the avatar moves: some never move, some lunge at an avatar that
stops right in front of them, some walk on and turn back at dead ends, some wake when the
avatar stands two nodes in front of them and then walk its trail two moves behind. A
watcher that lands on the avatar ends the game; the avatar that moves onto a watcher
removes it. Every key press spends a step of the level's budget.

From the frames alone:

* eyed blocks are squares of one colour with one pixel of another in the middle of a side
  (the way they face); the avatar is the one the keys move, and its look is kept for the
  game;
* the lattice has a node every two block widths from the avatar and a link where the
  block between two nodes is drawn; the exit is the solid block of a colour of its own on a
  node;
* what each kind of watcher (by its colour) does is learned by watching it: it moved
  without waking (it walks), its eye changed and it then moved (it trails), it landed on the
  avatar from the node in front of it (it lunges). Until a kind is known, the node in front
  of it is never stopped on. That moving onto a watcher removes it is learned the first
  time a plan does it (until then it is priced above a step).

The plan: a search over the avatar's node and the watchers' states, turn by turn, for the
fewest keys to the exit, never ending a turn where a watcher would land.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

Action = tuple[GameAction, dict[str, int] | None]

KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
MAX_NODES = 300_000
MAX_DEATHS = 4
MAX_LEVEL_ACTIONS = 300
STOMP_PRICE = 3  # moving onto a watcher, before it is known to remove it
MAX_WATCHES = 6  # safe moves made only to see what the watchers do


def add(a, b, k: int = 1):
    return (a[0] + b[0] * k, a[1] + b[1] * k)


def neg(d):
    return (-d[0], -d[1])


# --- reading the screen ------------------------------------------------------------


def eyed_blocks(grid: np.ndarray) -> list:
    """(body colour, eye colour, x, y, size, facing) of every square of one colour with one
    pixel of another in the middle of a side."""
    out = []
    for c in fp.segment(grid):
        s = c.width
        if c.height != s or s < 3 or s % 2 == 0 or s > 7 or c.area != s * s - 1:
            continue
        m = s // 2
        box = {(c.y1 + a, c.x1 + b) for a in range(s) for b in range(s)}
        own = {(int(p[0]), int(p[1])) for p in c.pixels}
        hole = (box - own).pop()
        dy, dx = hole[0] - c.y1, hole[1] - c.x1
        facing = {(0, m): (0, -1), (s - 1, m): (0, 1), (m, 0): (-1, 0), (m, s - 1): (1, 0)}.get((dy, dx))
        if facing is None:
            continue
        eye = int(grid[hole])
        if eye == c.color:
            continue
        out.append((c.color, eye, c.x1, c.y1, s, facing))
    return out


class Maze:
    """Nodes on a lattice and the links between them, in lattice coordinates."""

    def __init__(self, grid: np.ndarray, x0: int, y0: int, s: int) -> None:
        self.s, self.p = s, 2 * s
        self.ox, self.oy = x0 % self.p, y0 % self.p
        counts = Counter(int(v) for v in grid[:63].ravel())
        self.bg = counts.most_common(1)[0][0]
        self.nodes: set = set()
        for j in range(64 // self.p + 1):
            for i in range(64 // self.p + 1):
                b = self.block(grid, (i, j))
                if b is not None and not (b == self.bg).all():
                    self.nodes.add((i, j))
        self.links: dict = {n: set() for n in self.nodes}
        for n in self.nodes:
            for d in KEYS.values():
                m = add(n, d)
                if m not in self.nodes:
                    continue
                x, y = self.ox + n[0] * self.p + d[0] * s, self.oy + n[1] * self.p + d[1] * s
                b = grid[y:y + s, x:x + s]
                if b.shape == (s, s) and not (b == self.bg).any():
                    self.links[n].add(d)

    def block(self, grid: np.ndarray, n):
        x, y = self.ox + n[0] * self.p, self.oy + n[1] * self.p
        if x < 0 or y < 0 or x + self.s > 64 or y + self.s > 63:
            return None
        return grid[y:y + self.s, x:x + self.s]

    def node_of(self, x: int, y: int):
        if (x - self.ox) % self.p or (y - self.oy) % self.p:
            return None
        return ((x - self.ox) // self.p, (y - self.oy) // self.p)


class Watcher:
    def __init__(self, colour: int, eye: int, node, facing) -> None:
        self.colour, self.eye, self.node, self.facing = colour, eye, node, facing
        self.awake = False
        self.queue: tuple = ()

    def key(self) -> tuple:
        return (self.colour, self.node, self.facing, self.awake, self.queue)


# --- the turn ----------------------------------------------------------------------


def turn(maze: Maze, kinds: dict, a, d, ws: tuple):
    """The avatar at a moves along d (a link): (avatar, watchers) after the turn, or None if
    a watcher lands on it (or may, for a kind not known)."""
    q = add(a, d)
    # the avatar arrives: a watcher there is removed
    ws = tuple(w for w in ws if w[1] != q)
    out = []
    for colour, n, f, awake, queue in ws:
        k = kinds.get(colour)
        if k == "walk":
            n = add(n, f)
            if n == q:
                return None
            if f not in maze.links.get(n, ()):
                f = neg(f)
        elif k == "trail":
            if awake:
                queue = queue + (d,)
                n = add(n, f)
                if n == q:
                    return None
                f, queue = queue[0], queue[1:]
            elif add(n, f, 2) == q:
                awake, queue = True, (f,)
        else:
            # never seen moving (it may lunge): its front is no place to stop
            if add(n, f) == q:
                return None
        out.append((colour, n, f, awake, queue))
    return q, tuple(sorted(out))


def search(maze: Maze, kinds: dict, a, ws: tuple, goal, stomp_known: bool, limit: int):
    """The fewest keys to the goal: list of moves (directions), or None."""
    start = (a, ws)
    prev = {start: None}
    frontier = deque([(start, 0)])
    # stomps cost more while not known: a two-queue walk (0-1 BFS with small weights)
    best = {start: 0}
    buckets: dict = {0: deque([start])}
    cost = 0
    while buckets:
        if cost not in buckets:
            cost = min(buckets)
        q = buckets[cost]
        if not q:
            del buckets[cost]
            continue
        st = q.popleft()
        if best.get(st, 1 << 30) < cost:
            continue
        a, ws = st
        if a == goal:
            path = []
            while prev[st] is not None:
                st, d = prev[st]
                path.append(d)
            return path[::-1]
        if cost >= limit or len(best) > MAX_NODES:
            continue
        for d in maze.links.get(a, ()):
            r = turn(maze, kinds, a, d, ws)
            if r is None:
                continue
            step = 1
            if not stomp_known and any(w[1] == add(a, d) for w in ws):
                step += STOMP_PRICE
            nc = cost + step
            if nc < best.get(r, 1 << 30):
                best[r] = nc
                prev[r] = (st, d)
                buckets.setdefault(nc, deque()).append(r)
    return None


# --- the solver --------------------------------------------------------------------


class MazeSolver:
    name = "maze"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        self.kinds: dict = {}  # watcher colour -> walk / trail / lunge / still
        self.avatar_look: tuple | None = None  # (body, eye)
        self.stomp_known = False
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.read = False
        self.maze: Maze | None = None
        self.queue: list = []
        self.expect = None
        self.deaths = 0
        self.actions = 0
        self.fails = 0
        self.watches = 0

    def _give_up(self) -> None:
        self.gave_up = True
        self.in_control = False
        self.queue = []
        self.expect = None

    def _read(self, grid: np.ndarray) -> bool:
        """The maze, the avatar, the exit and the watchers, from a frame."""
        blocks = eyed_blocks(grid)
        if not blocks:
            return False
        if self.avatar_look is not None:
            mine = [b for b in blocks if (b[0], b[1]) == self.avatar_look]
        else:
            # the one eyed block whose eye is of a colour no other has
            eyes = Counter(b[1] for b in blocks)
            mine = [b for b in blocks if eyes[b[1]] == 1]
            if len(mine) > 1:
                mine = mine[:1] if len(blocks) == 1 else [b for b in mine if Counter(x[0] for x in blocks)[b[0]] == 1][:1]
        if len(mine) != 1:
            return False
        body, eye, x, y, s, f = mine[0]
        maze = Maze(grid, x, y, s)
        a = maze.node_of(x, y)
        if a is None or not maze.links.get(a):
            return False
        ws = []
        for b in blocks:
            if b is mine[0] or b[4] != s:
                continue
            n = maze.node_of(b[2], b[3])
            if n is None:
                continue
            ws.append(Watcher(b[0], b[1], n, b[5]))
        # the exit: a solid block of a colour of its own on a node
        node_colours = Counter()
        solid = {}
        for n in maze.nodes:
            blk = maze.block(grid, n)
            vals = np.unique(blk)
            if len(vals) == 1:
                node_colours[int(vals[0])] += 1
                solid[n] = int(vals[0])
        taken = {a} | {w.node for w in ws}
        lone = [n for n, c in solid.items() if node_colours[c] == 1 and n not in taken and c not in (body, eye)]
        if len(lone) != 1:
            return False
        self.maze, self.avatar, self.goal, self.watchers = maze, a, lone[0], ws
        self.body = (body, eye)
        return True

    def _watchers_now(self, grid: np.ndarray) -> list:
        m = self.maze
        out = []
        for b in eyed_blocks(grid):
            if (b[0], b[1]) == self.body or b[4] != m.s:
                continue
            n = m.node_of(b[2], b[3])
            if n is not None:
                out.append(Watcher(b[0], b[1], n, b[5]))
        return out

    def _avatar_now(self, grid: np.ndarray):
        m = self.maze
        for b in eyed_blocks(grid):
            if (b[0], b[1]) == self.body:
                return m.node_of(b[2], b[3])
        return None

    def _learn(self, before: list, after: list, a_before, a_after) -> None:
        """What each watcher did this turn says what its kind does."""
        left = list(after)
        for w in before:
            if w.node == a_after:
                continue  # removed by the avatar
            near = [v for v in left if v.colour == w.colour and abs(v.node[0] - w.node[0]) + abs(v.node[1] - w.node[1]) <= 1]
            if not near:
                continue
            v = min(near, key=lambda v: (v.node != add(w.node, w.facing), v.node != w.node))
            left.remove(v)
            woke = w.eye != -1 and v.eye != w.eye
            if v.node == add(w.node, w.facing):
                k = "trail" if (w.awake or woke) else "walk"
                if self.kinds.get(w.colour) != k:
                    self.kinds[w.colour] = k
                    self.stats[f"learned_{k}"] += 1
            if woke and v.node == w.node and self.kinds.get(w.colour) is None:
                self.kinds[w.colour] = "trail"
                self.stats["learned_trail"] += 1
        if a_after is not None and any(w.node == a_after for w in before):
            if not self.stomp_known:
                self.stomp_known = True
                self.stats["learned_stomp"] += 1

    def _state(self) -> tuple:
        return tuple(sorted(w.key() for w in self.watchers))

    def _plan(self) -> list:
        keys = {d: k for k, d in KEYS.items()}
        path = search(self.maze, self.kinds, self.avatar, self._state(), self.goal, self.stomp_known, 200)
        if path is not None:
            return [keys[d] for d in path]
        # no way known: one safe move, to watch the watchers act (towards the exit if it can)
        if self.watches >= MAX_WATCHES:
            return []
        state = self._state()
        safe = [d for d in self.maze.links.get(self.avatar, ()) if turn(self.maze, self.kinds, self.avatar, d, state) is not None
                and not any(w[1] == add(self.avatar, d) for w in state)]
        if not safe:
            return []
        far = self._distances(self.goal)
        d = min(safe, key=lambda d: far.get(add(self.avatar, d), 1 << 20))
        self.watches += 1
        self.stats["watch_moves"] += 1
        return [keys[d]]

    def _distances(self, goal) -> dict:
        dist = {goal: 0}
        q = deque([goal])
        while q:
            n = q.popleft()
            for d in self.maze.links.get(n, ()):
                m = add(n, d)
                if m not in dist:
                    dist[m] = dist[n] + 1
                    q.append(m)
        return dist

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            if obs.state == GameState.GAME_OVER and self.expect is not None and self.maze is not None:
                self._died()
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        exp, self.expect = self.expect, None
        if action is None or self.maze is None:
            return
        grid = np.asarray(obs.frame[-1])
        if action is GameAction.RESET or exp is None:
            # read the level again (after a reset, or a step not ours)
            self.maze = None
            self.read = False
            self.queue = []
            return
        a_before, ws_before, k = exp
        a_now = self._avatar_now(grid)
        ws_now = self._watchers_now(grid)
        if a_now is None:
            self._fail("avatar_lost")
            return
        d = KEYS[k]
        if a_now == a_before:
            # the key did not move it: no link that way after all
            self.maze.links[a_before].discard(d)
            self._fail("blocked")
            return
        if a_now != add(a_before, d):
            self._fail("moved_elsewhere")
            self.avatar = a_now
            self.watchers = ws_now
            return
        if self.avatar_look is None:
            self.avatar_look = self.body
        state_before = tuple(sorted(w.key() for w in ws_before))
        predicted = turn(self.maze, self.kinds, a_before, d, state_before)
        self.avatar = a_now
        seen = {(v.colour, v.node, v.facing) for v in ws_now}
        eyes_before = {(w.colour, w.node, w.facing): w.eye for w in ws_before}
        woke = any(eyes_before.get((v.colour, v.node, v.facing), -1) not in (-1, v.eye) for v in ws_now)
        if predicted is not None and not woke and self._agrees(predicted[1], seen):
            # as the model said: keep its states (who is awake, the trails they walk)
            eyes = {(v.colour, v.node, v.facing): v.eye for v in ws_now}
            self.watchers = []
            for c, n, f, awake, queue in predicted[1]:
                w = Watcher(c, eyes.get((c, n, f), -1), n, f)
                w.awake, w.queue = awake, queue
                self.watchers.append(w)
            return
        self._learn(ws_before, ws_now, a_before, a_now)
        # what is seen, with what is known of each watcher carried over
        for v in ws_now:
            old = [w for w in ws_before if w.colour == v.colour and abs(w.node[0] - v.node[0]) + abs(w.node[1] - v.node[1]) <= 1]
            if old:
                w = min(old, key=lambda w: abs(w.node[0] - v.node[0]) + abs(w.node[1] - v.node[1]))
                v.awake = w.awake or (w.eye != -1 and v.eye != w.eye)
                if v.awake:
                    # on the trail it holds the avatar's latest move; just woken, its own way
                    v.queue = (d,) if w.awake else (v.facing,)
        self.watchers = ws_now
        if predicted is not None:
            self.stats["mismatch"] += 1
        self.queue = []

    @staticmethod
    def _agrees(predicted: tuple, seen: set) -> bool:
        """Every watcher seen was predicted, and every one predicted but not seen shares its
        node with another (drawn over it)."""
        mine = {(c, n, f) for c, n, f, _, _ in predicted}
        if not seen <= mine:
            return False
        nodes = Counter(n for _, n, _, _, _ in predicted)
        return all((c, n, f) in seen or nodes[n] > 1 for c, n, f, _, _ in predicted)

    def _died(self) -> None:
        a_before, ws_before, k = self.expect
        q = add(a_before, KEYS[k])
        for w in ws_before:
            if add(w.node, w.facing) == q and self.kinds.get(w.colour) in (None, "still"):
                self.kinds[w.colour] = "lunge"
                self.stats["learned_lunge"] += 1
        self.deaths += 1
        self.stats["deaths"] += 1
        self.expect = None
        self.queue = []

    def _fail(self, why: str) -> None:
        self.fails += 1
        self.stats[why] += 1
        self.queue = []

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4} <= set(obs.available_actions):
            return None
        if self.deaths >= MAX_DEATHS or self.fails >= 6 or self.actions >= MAX_LEVEL_ACTIONS:
            self.stats["gave_up"] += 1
            self._give_up()
            return None
        grid = np.asarray(obs.frame[-1])
        if self.maze is None:
            if self.read:
                return None
            self.read = True
            if not self._read(grid):
                self.gave_up = True  # not this family's screen (looked at once a level)
                return None
        if not self.queue:
            self.queue = self._plan()
            if not self.queue:
                self.stats["no_plan"] += 1
                self._give_up()
                return None
            if not self.in_control:
                self.in_control = True
                self.stats["takeovers"] += 1
        k = self.queue.pop(0)
        self.expect = (self.avatar, [w for w in self.watchers], k)
        self.actions += 1
        self.stats["actions"] += 1
        return k, None
