"""The meet solver on a small game: two bodies on a grid of 4-pixel cells; every key moves
both a cell, the second one mirrored left to right; a wall stops one body and not the
other; the level is done when they land on one cell or cross. On the second level a gate
in a wall opens while a body stands on its switch (a lone cell of the gate's colour), and
traps (checkered cells) send everything back to the start. On the third a block sits in
the corridor between the bodies: a click picks it out, the keys then move it, a click on
the floor lets it go. Colours and layout are none of m0r0's. Nothing in the solver knows
which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

FLOOR, WALL, BODY, BODY_SEL, TRAP, GATE, CORE, CORE_SEL = 3, 7, 12, 1, 13, 9, 14, 11
C, N, OFF = 4, 12, 8
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


class MeetRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        border = {(x, y) for x in range(N) for y in range(N) if x in (0, N - 1) or y in (0, N - 1)}
        self.walls, self.traps, self.gate, self.switch, self.blocks = set(border), set(), set(), set(), []
        self.selected = None
        if self.done == 0:
            self.start = [(2, 8), (9, 3)]
        elif self.done == 1:
            self.walls |= {(6, y) for y in range(1, N - 1) if y != 6} | {(2, 6)}
            self.gate = {(6, 6)}
            self.switch = {(1, 6)}
            self.traps = {(3, 2), (9, 2)}
            self.start = [(3, 3), (9, 3)]
        else:
            self.walls |= {(x, y) for x in range(1, N - 1) for y in (4, 6) if (x, y) != (6, 4)}
            self.blocks = [(6, 5)]
            self.start = [(2, 5), (9, 5)]
        self.bodies = list(self.start)
        self.start_blocks = list(self.blocks)

    def _gate_open(self) -> bool:
        return any(b in self.switch for b in self.bodies)

    def _stops(self, t, gate_open) -> bool:
        return t in self.walls or t in self.blocks or (t in self.gate and not gate_open)

    def _step_bodies(self, d) -> None:
        gate_open = self._gate_open()
        new = []
        for i, b in enumerate(self.bodies):
            dd = (-d[0], d[1]) if i == 1 else d
            t = add(b, dd)
            new.append(b if self._stops(t, gate_open) else t)
        if any(t in self.traps for t in new):
            self.bodies, self.blocks = list(self.start), list(self.start_blocks)
            return
        a0, b0 = self.bodies
        a1, b1 = new
        self.bodies = new
        if a1 == b1 or (a1 == b0 and b1 == a0):
            self.won = True

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), WALL, dtype=np.int8)

        def fill(c, colour):
            x, y = c
            g[OFF + y * C:OFF + (y + 1) * C, OFF + x * C:OFF + (x + 1) * C] = colour

        for x in range(N):
            for y in range(N):
                fill((x, y), WALL if (x, y) in self.walls else FLOOR)
        for c in self.traps:
            x, y = c
            cell = np.full((C, C), FLOOR, dtype=np.int8)
            cell[0::2, 0::2] = TRAP
            cell[1::2, 1::2] = TRAP
            g[OFF + y * C:OFF + (y + 1) * C, OFF + x * C:OFF + (x + 1) * C] = cell
        open_ = self._gate_open()
        for c in self.gate:
            fill(c, FLOOR if open_ else GATE)
        for c in self.switch:
            fill(c, GATE)
        for c in self.blocks:
            x, y = c
            fill(c, FLOOR)
            core = CORE_SEL if c == self.selected else CORE
            g[OFF + y * C + 1:OFF + (y + 1) * C - 1, OFF + x * C + 1:OFF + (x + 1) * C - 1] = core
        for b in self.bodies:
            fill(b, BODY_SEL if self.selected is not None else BODY)
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, 3, frames, [1, 2, 3, 4, 6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        self.won = False
        if action is GameAction.ACTION6:
            x, y = (data["x"] - OFF) // C, (data["y"] - OFF) // C
            self.selected = (x, y) if (x, y) in self.blocks else None
        elif action in KEYS:
            d = KEYS[action]
            if self.selected is not None:
                t = add(self.selected, d)
                if not (t in self.walls or t in self.blocks or t in self.bodies or t in self.traps or t in self.gate):
                    self.blocks[self.blocks.index(self.selected)] = t
                    self.selected = t
            else:
                self._step_bodies(d)
        if self.won:
            finished = self._grid()
            self.done += 1
            if self.done == 3:
                self.state = GameState.WIN
                return self._obs([finished])
            self._load()
            return self._obs([finished, self._grid()])
        return self._obs([self._grid()])


def test_meet_solver_brings_mirrored_bodies_together_past_gates_traps_and_blocks():
    env = MeetRoom()
    agent = LibraryAgent(seed=0)
    obs = env.reset()
    done_at = []
    for used in range(1, 300):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(done_at) < obs.levels_completed:
            done_at.append(used)
        if obs.state == GameState.WIN:
            break
    solver = next(s for s in agent.solvers if s.name == "meet")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    assert per[0] <= 16, (per, dict(solver.stats))
    assert per[1] <= 40, (per, dict(solver.stats))
    assert per[2] <= 20, (per, dict(solver.stats))
    for k in ("mirror_learned", "traps_learned", "switches_learned", "movable_learned"):
        assert solver.stats[k] >= 1, (k, dict(solver.stats))
