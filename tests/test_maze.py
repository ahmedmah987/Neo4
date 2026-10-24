"""The maze solver on a small game: nodes joined by links on a lattice of 5-pixel blocks,
an avatar with an eye on the side it faces, an exit. On the second level a watcher lunges
at an avatar that stops in front of it (on the short way) and another walks up and down
the column of the exit, turning at the ends; on the third, a watcher wakes when the
avatar stands two nodes in front of it and then walks the avatar's trail two moves behind.
A watcher landing on the avatar ends the game. Colours, block size and layout are none of
tu93's, and the rules here are written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, NODE, LINK, BODY, EYE, EXIT, WEYE, AWAKE = 3, 1, 7, 10, 12, 6, 0, 2
KINDS = {"lunge": 14, "walk": 11, "trail": 13}
S, P, OFF, N = 5, 10, 2, 6
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def add(a, b, k=1):
    return (a[0] + b[0] * k, a[1] + b[1] * k)


def path_links(cells):
    out = set()
    for a, b in zip(cells, cells[1:]):
        out.add(frozenset((a, b)))
    return out


class MazeRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        if self.done == 0:
            snake = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4),
                     (3, 4), (4, 4), (5, 4), (5, 5)]
            self.links = path_links(snake) | path_links([(2, 0), (3, 0), (4, 0)]) | path_links([(2, 2), (3, 2), (3, 3)])
            self.avatar, self.exit, self.watchers = (0, 0), (5, 5), []
        elif self.done == 1:
            rows = [[(x, 0) for x in range(6)], [(x, 3) for x in range(6)]]
            cols = [[(0, y) for y in range(4)], [(5, y) for y in range(4)]]
            self.links = set().union(*(path_links(r) for r in rows + cols))
            self.avatar, self.exit = (0, 0), (5, 3)
            self.extra_nodes = {(3, 1)}
            self.watchers = [["lunge", (3, 1), (0, -1), False, []], ["walk", (5, 1), (0, 1), False, []]]
        else:
            self.links = path_links([(x, 2) for x in range(5)]) | path_links([(3, 0), (3, 1), (3, 2)]) | path_links([(4, 2), (4, 3), (4, 4)])
            self.avatar, self.exit = (0, 2), (4, 4)
            self.watchers = [["trail", (3, 0), (0, 1), False, []]]
        self.facing = (1, 0)
        self.nodes = {c for l in self.links for c in l} | getattr(self, "extra_nodes", set())
        self.extra_nodes = set()

    def _linked(self, a, d) -> bool:
        return frozenset((a, add(a, d))) in self.links

    def _turn(self, d) -> bool:
        """The avatar moves along d; False if a watcher lands on it."""
        q = add(self.avatar, d)
        self.avatar = q
        self.watchers = [w for w in self.watchers if w[1] != q]
        for w in self.watchers:
            kind, n, f, awake, queue = w
            if kind == "lunge" and add(n, f) == q:
                return False
            if kind == "walk":
                n = add(n, f)
                if n == q:
                    return False
                if not self._linked(n, f):
                    f = (-f[0], -f[1])
                w[1], w[2] = n, f
            if kind == "trail":
                if awake:
                    queue.append(d)
                    n = add(n, f)
                    if n == q:
                        return False
                    w[1] = n
                    w[2] = queue.pop(0)
                elif add(n, f, 2) == q:
                    w[3] = True
                    w[4] = [f]
        return True

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)

        def block(c, colour):
            x, y = OFF + c[0] * P, OFF + c[1] * P
            g[y:y + S, x:x + S] = colour

        for c in self.nodes:
            block(c, NODE)
        for l in self.links:
            a, b = sorted(l)
            x, y = OFF + a[0] * P + (b[0] - a[0]) * S, OFF + a[1] * P + (b[1] - a[1]) * S
            g[y:y + S, x:x + S] = LINK
        block(self.exit, EXIT)

        def eyed(c, body, eye, f):
            block(c, body)
            x, y = OFF + c[0] * P, OFF + c[1] * P
            m = S // 2
            ex, ey = {(0, -1): (m, 0), (0, 1): (m, S - 1), (-1, 0): (0, m), (1, 0): (S - 1, m)}[f]
            g[y + ey, x + ex] = eye

        for kind, n, f, awake, _ in self.watchers:
            eyed(n, KINDS[kind], AWAKE if awake else WEYE, f)
        eyed(self.avatar, BODY, EYE, self.facing)
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, 3, frames, [1, 2, 3, 4])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        if action in KEYS:
            d = KEYS[action]
            if self._linked(self.avatar, d):
                self.facing = d
                if not self._turn(d):
                    self.state = GameState.GAME_OVER
                    return self._obs([])
                if self.avatar == self.exit:
                    finished = self._grid()
                    self.done += 1
                    if self.done == 3:
                        self.state = GameState.WIN
                        return self._obs([finished])
                    self._load()
                    return self._obs([finished, self._grid()])
        return self._obs([self._grid()])


def test_maze_solver_walks_past_lungers_walkers_and_trailers():
    env = MazeRoom()
    agent = LibraryAgent(seed=0)
    obs = env.reset()
    done_at = []
    for used in range(1, 200):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(done_at) < obs.levels_completed:
            done_at.append(used)
        if obs.state == GameState.WIN:
            break
    solver = next(s for s in agent.solvers if s.name == "maze")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    assert per[0] <= 14, (per, dict(solver.stats))
    assert per[1] <= 14, (per, dict(solver.stats))
    assert per[2] <= 6, (per, dict(solver.stats))
    assert solver.stats["deaths"] == 0, dict(solver.stats)
    assert solver.stats["learned_walk"] >= 1 and solver.stats["learned_trail"] >= 1, dict(solver.stats)
