"""The carry solver on a small game: a player on a grid of 4-pixel cells (one edge drawn
in another colour: the side it faces), framed blocks, a goal zone outlined round its
inside colour. Keys 1-4 move the player a cell and turn it that way; key 5 picks up the
block it faces or puts down the one it carries, which stays in front of it. On the
second level a dotted barrier, which stops walkers but not carried blocks, keeps the
player away from the goal, and a helper on the other side fetches free blocks into it
(the player must put its blocks down where the helper can reach them). Colours and
layout are none of wa30's. Nothing in the solver knows which game it plays."""

from collections import deque

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

FLOOR, BODY, EDGE, RING, CORE, OUTLINE, INSIDE, WALL, DOTS, HELPER = 3, 6, 8, 5, 11, 12, 10, 2, 9, 13
C, N = 4, 16
DIRS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


class CarryRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        self.facing = (0, -1)
        self.carry = None  # offset of the carried block from the player
        self.helper_carry = None
        if self.done == 0:
            self.player = (2, 10)
            self.blocks = {(5, 5), (9, 7)}
            self.goal = {(12, 3), (13, 3)}
            self.walls, self.barrier, self.helper = set(), set(), None
        else:
            self.player = (3, 8)
            self.blocks = {(4, 4), (5, 11), (11, 12)}  # the last one on the helper's side
            self.goal = {(12, 7), (13, 7), (12, 8), (13, 8)}
            self.walls = set()
            self.barrier = {(8, y) for y in range(N)}
            self.helper = (11, 2)

    # --- rules -----------------------------------------------------------------------

    def _inside(self, c):
        return 0 <= c[0] < N and 0 <= c[1] < N

    def _free_walk(self, c, ignore=()):
        return self._inside(c) and c not in self.walls and c not in self.barrier and (
            c in ignore or (c not in self.blocks and c != self.player and c != self.helper))

    def _free_block(self, c, ignore=()):
        return self._inside(c) and c not in self.walls and (
            c in ignore or (c not in self.blocks and c != self.player and c != self.helper))

    def _move_pair(self, who, off, d):
        b = add(who, off)
        nw, nb = add(who, d), add(b, d)
        if self._free_walk(nw, ignore={b}) and self._free_block(nb, ignore={who}):
            self.blocks.discard(b)
            self.blocks.add(nb)
            return nw
        return who

    def _helper_step(self) -> None:
        h = self.helper
        if h is None:
            return
        if self.helper_carry is not None:
            b = add(h, self.helper_carry)
            if b in self.goal:
                self.helper_carry = None
                return
            path = self._bfs(h, lambda c: add(c, self.helper_carry) in self.goal and add(c, self.helper_carry) not in self.blocks - {b},
                             lambda c, n: self._free_walk(n, ignore={b}) and self._free_block(add(n, self.helper_carry), ignore={c, b}))
            if path:
                self.helper = self._move_pair(h, self.helper_carry, path[0])
            return
        free = [b for b in self.blocks if b not in self.goal and not (self.carry and b == add(self.player, self.carry))]
        for b in free:
            if abs(b[0] - h[0]) + abs(b[1] - h[1]) == 1:
                self.helper_carry = (b[0] - h[0], b[1] - h[1])
                return
        targets = {add(b, d) for b in free for d in DIRS.values()}
        path = self._bfs(h, lambda c: c in targets, lambda c, n: self._free_walk(n))
        if path:
            self.helper = add(h, path[0])

    @staticmethod
    def _bfs(start, goal_fn, ok):
        seen = {start}
        q = deque([(start, [])])
        while q:
            c, path = q.popleft()
            if path and goal_fn(c):
                return path
            for d in DIRS.values():
                n = add(c, d)
                if n not in seen and ok(c, n):
                    seen.add(n)
                    q.append((n, path + [d]))
        return None

    def _won(self) -> bool:
        return self.blocks <= self.goal and self.carry is None and self.helper_carry is None

    # --- drawing ---------------------------------------------------------------------

    def _grid(self) -> np.ndarray:
        g = np.full((N * C, N * C), FLOOR, dtype=np.int8)
        xs = [x for x, _ in self.goal]
        ys = [y for _, y in self.goal]
        x1, x2, y1, y2 = min(xs) * C, (max(xs) + 1) * C, min(ys) * C, (max(ys) + 1) * C
        g[y1:y2, x1:x2] = OUTLINE
        g[y1 + 1:y2 - 1, x1 + 1:x2 - 1] = INSIDE
        for x, y in self.walls:
            g[y * C:(y + 1) * C, x * C:(x + 1) * C] = WALL
        for x, y in self.barrier:
            cell = np.full((C, C), DOTS, dtype=np.int8)
            cell[(y % 2)::2, ::2] = FLOOR
            g[y * C:(y + 1) * C, x * C:(x + 1) * C] = cell
        for x, y in self.blocks:
            g[y * C:(y + 1) * C, x * C:(x + 1) * C] = RING
            g[y * C + 1:(y + 1) * C - 1, x * C + 1:(x + 1) * C - 1] = CORE
        if self.helper is not None:
            x, y = self.helper
            g[y * C:(y + 1) * C, x * C:(x + 1) * C] = HELPER
        x, y = self.player
        cell = np.full((C, C), BODY, dtype=np.int8)
        dx, dy = self.facing
        if dy < 0:
            cell[0] = EDGE
        elif dy > 0:
            cell[-1] = EDGE
        elif dx < 0:
            cell[:, 0] = EDGE
        else:
            cell[:, -1] = EDGE
        g[y * C:(y + 1) * C, x * C:(x + 1) * C] = cell
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, 2, frames, [1, 2, 3, 4, 5])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        if action in DIRS:
            d = DIRS[action]
            if self.carry is None:
                self.facing = d
                if self._free_walk(add(self.player, d)):
                    self.player = add(self.player, d)
            else:
                self.player = self._move_pair(self.player, self.carry, d)
        elif action is GameAction.ACTION5:
            if self.carry is not None:
                self.carry = None
            else:
                front = add(self.player, self.facing)
                held = add(self.helper, self.helper_carry) if self.helper_carry else None
                if front in self.blocks and front != held:
                    self.carry = self.facing
        self._helper_step()
        if self._won():
            finished = self._grid()
            self.done += 1
            if self.done == 2:
                self.state = GameState.WIN
                return self._obs([finished])
            self._load()
            return self._obs([finished, self._grid()])
        return self._obs([self._grid()])


def test_carry_solver_brings_the_blocks_in_and_hands_them_over_the_barrier():
    env = CarryRoom()
    agent = LibraryAgent(seed=0)
    obs = env.reset()
    done_at = []
    for used in range(1, 400):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(done_at) < obs.levels_completed:
            done_at.append(used)
        if obs.state == GameState.WIN:
            break
    solver = next(s for s in agent.solvers if s.name == "carry")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    assert per[0] <= 40, (per, dict(solver.stats))
    assert per[1] <= 80, (per, dict(solver.stats))
    assert solver.stats["learned_blocking"] >= 1, dict(solver.stats)
