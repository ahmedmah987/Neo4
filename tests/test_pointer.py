"""The pointer-family solver on a small game: two handles, a body that sits at
their centroid, and a ring of dots of the body's colour. Clicking a handle
selects it; clicking a free place sends the selected handle's centre there. The
level ends when the body is centred on the ring; nothing in the solver knows
which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG = 5
CROSS = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2)]


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


class HandleRoom:
    LAYOUTS = [  # (handle centres, ring centre)
        ([(10, 40), (30, 50)], (40, 20)),
        ([(12, 12), (50, 14)], (20, 44)),
    ]

    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _place(self) -> None:
        handles, self.ring = self.LAYOUTS[self.done % len(self.LAYOUTS)]
        self.handles = [list(h) for h in handles]
        self.sel = 0

    def _body(self) -> tuple[int, int]:
        return (sum(h[0] for h in self.handles) // 2, sum(h[1] for h in self.handles) // 2)

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)
        rx, ry = self.ring
        for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, 3), (-3, 3), (3, -3)):
            g[ry + dy, rx + dx] = 15
        bx, by = self._body()
        g[by - 2:by + 3, bx - 2:bx + 3] = 15
        for i, (hx, hy) in enumerate(self.handles):
            for dy, dx in CROSS:
                g[hy - 2 + dy, hx - 2 + dx] = 0 if i == self.sel else 3
            g[hy, hx] = 15
        return g

    def _obs(self, frames=None) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(self.LAYOUTS), (frames or []) + [self._grid()], [6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._place()
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action is not GameAction.ACTION6 or data is None:
            return self._obs()
        x, y = data["x"], data["y"]
        for i, (hx, hy) in enumerate(self.handles):
            if abs(x - hx) <= 2 and abs(y - hy) <= 2:
                self.sel = i
                return self._obs()
        if 2 <= x <= 61 and 2 <= y <= 61:
            self.handles[self.sel] = [x, y]
        if self._body() == self.ring:
            finished = self._grid()
            self.done += 1
            if self.done == len(self.LAYOUTS):
                self.state = GameState.WIN
            else:
                self._place()
            return self._obs([finished])
        return self._obs()


def test_pointer_solver_centres_the_body_on_its_ring():
    env = HandleRoom()
    agent = LibraryAgent(seed=0)
    obs = env.reset()
    done_at = []
    for used in range(1, 600):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(done_at) < obs.levels_completed:
            done_at.append(used)
        if obs.state == GameState.WIN:
            break
    assert len(done_at) == 2, (done_at, agent.stats)
    pointer = next(s for s in agent.solvers if s.name == "pointer")
    assert pointer.stats["gotos"] >= 2
    # the second level is planned outright: a few clicks (a handle may not go onto
    # the ring's dots, so two handles may move), against the first level's exploring
    assert done_at[1] - done_at[0] <= 8, (done_at, dict(pointer.stats))
