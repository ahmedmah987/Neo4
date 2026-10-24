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


DISC = [(dy, dx) for dy in range(5) for dx in range(5) if (dy, dx) not in ((0, 0), (0, 4), (4, 0), (4, 4))]
BASE = 7


class PaintRoom:
    """A blank body between two handles; pots of the body's shape paint half of it
    (left in colour 8, right in 9, top in 12) and vanish when it lands on them; the
    level ends when the body, painted in exactly 8 and 9, is centred on the ring
    drawn in those colours. Another ring, in 12, is a decoy."""

    HANDLES = [(12, 40), (28, 50)]
    POTS = {(40, 12): (8, "left"), (12, 14): (9, "right"), (44, 44): (12, "top")}
    GOAL, DECOY = (50, 28), (30, 30)

    def __init__(self) -> None:
        self.state = GameState.NOT_PLAYED

    @staticmethod
    def _half(side: str, dy: int, dx: int) -> bool:
        return {"left": dx <= 1, "right": dx >= 3, "top": dy <= 1}[side]

    def _body(self) -> tuple[int, int]:  # top-left
        cx = sum(h[0] for h in self.handles) // 2
        cy = sum(h[1] for h in self.handles) // 2
        return cx - 2, cy - 2

    def _ring(self, g, centre, colour_of) -> None:
        """A ring of dots round a 7x7 square, its dots a pixel apart."""
        cx, cy = centre
        for dy, dx in ((0, 2), (0, 4), (1, 1), (1, 5), (2, 0), (2, 6), (4, 0), (4, 6), (5, 1), (5, 5), (6, 2), (6, 4)):
            g[cy - 3 + dy, cx - 3 + dx] = colour_of(dx - 3)

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)
        self._ring(g, self.GOAL, lambda dx: 8 if dx <= 0 else 9)
        self._ring(g, self.DECOY, lambda dx: 12)
        for (px, py), (col, side) in self.pots.items():
            for dy, dx in DISC:
                g[py + dy, px + dx] = col if self._half(side, dy, dx) else BASE
        bx, by = self._body()
        for dy, dx in DISC:
            g[by + dy, bx + dx] = self.paint[(dy, dx)]
        for i, (hx, hy) in enumerate(self.handles):
            for dy, dx in CROSS:
                g[hy - 2 + dy, hx - 2 + dx] = 0 if i == self.sel else 3
            g[hy, hx] = 4
        return g

    def _obs(self, frames=None) -> FrameDataRaw:
        return obs_of(self.state, int(self.state == GameState.WIN), 1, (frames or []) + [self._grid()], [6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self.handles = [list(h) for h in self.HANDLES]
        self.sel = 0
        self.pots = dict(self.POTS)
        self.paint = {p: BASE for p in DISC}
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
        if not (2 <= x <= 61 and 2 <= y <= 61):
            return self._obs()
        self.handles[self.sel] = [x, y]
        bx, by = self._body()
        body = {(bx + dx, by + dy) for dy, dx in DISC}
        for (px, py), (col, side) in list(self.pots.items()):
            if body & {(px + dx, py + dy) for dy, dx in DISC}:
                for dy, dx in DISC:
                    if self._half(side, dy, dx):
                        self.paint[(dy, dx)] = col
                del self.pots[(px, py)]
        colours = set(self.paint.values()) - {BASE}
        if (bx + 2, by + 2) == self.GOAL and colours == {8, 9}:
            finished = self._grid()
            self.state = GameState.WIN
            return self._obs([finished])
        return self._obs()


def test_pointer_solver_paints_the_body_from_pots_before_its_goal():
    env = PaintRoom()
    agent = LibraryAgent(seed=0)
    obs = env.reset()
    for used in range(1, 600):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.state == GameState.WIN:
            break
    pointer = next(s for s in agent.solvers if s.name == "pointer")
    assert obs.state == GameState.WIN, (used, dict(pointer.stats), env.paint, env.pots)
    # it learned from the frames that the pots give colours, used the two the goal
    # needs and left the decoy pot alone
    assert pointer.absorb is True and pointer.stats["pot_taken"] >= 2
    assert (44, 44) in env.pots
    assert used <= 40, (used, dict(pointer.stats))
