"""The pour solver on a small game: a spout with a drop under it, bars, cups (U shapes) and
a floor on a grid of 4-pixel cells. A click picks a bar, the keys move it a cell, the fifth
key pours: water falls from the spout, runs along a bar it lands on and falls off both
ends, fills a cup it falls into, and the pour fails if any water reaches the floor. On the
second level the water rises (the spout is at the bottom), the bar picked out at the start
stands in the water's way, and another must be clicked and put across the stream. Colours
and layout are none of sp80's, and the flow here is written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, WATER, SPOUT, BAR, SEL, CUP, FILLED, FLOOR = 0, 10, 7, 2, 3, 5, 6, 9
C, N = 4, 16
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cup(x, y, down):
    """A U of five cells round the opening (x, y); down: it opens downwards."""
    side = 1 if not down else -1
    return {(x - 1, y), (x + 1, y), (x - 1, y + side), (x, y + side), (x + 1, y + side)}


class PourRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        if self.done == 0:
            self.g = (0, 1)
            self.spout = (8, 0)
            self.bars = [[(2, 6), 3]]
            self.cups = [(6, 13), (10, 13)]
            self.floor = {(x, 15) for x in range(N)}
        else:
            self.g = (0, -1)
            self.spout = (4, 15)
            self.bars = [[(7, 5), 3], [(1, 11), 5]]
            self.cups = [(2, 2), (8, 2)]
            self.floor = {(x, 0) for x in range(N)}
        self.selected = 0
        self.cup_cells = [cup(x, y, self.g == (0, -1)) for x, y in self.cups]

    def _bar_cells(self, i, at=None):
        (x, y), w = self.bars[i]
        if at is not None:
            x, y = at
        return {(x + k, y) for k in range(w)}

    def _solid(self):
        out = {self.spout, add(self.spout, self.g)} | self.floor
        for cs in self.cup_cells:
            out |= cs
        return out

    def _flow(self):
        bars = set().union(*(self._bar_cells(i) for i in range(len(self.bars))))
        cups = set().union(*self.cup_cells)
        wet, filled, floor = set(), set(), False
        streams, seen = [add(self.spout, self.g)], set()
        while streams:
            c = streams.pop()
            if c in seen:
                continue
            seen.add(c)
            wet.add(c)
            n = add(c, self.g)
            if not (0 <= n[0] < N and 0 <= n[1] < N):
                continue
            if n in self.floor:
                floor = True
                continue
            if n in cups:
                k = next(i for i, cs in enumerate(self.cup_cells) if n in cs)
                if (c[0] - 1, c[1]) in self.cup_cells[k] and (c[0] + 1, c[1]) in self.cup_cells[k]:
                    filled.add(k)
                    continue
            if n in bars or n in cups:
                # run along the top both ways and fall off the ends
                for side in (-1, 1):
                    t = c
                    while True:
                        t = (t[0] + side, t[1])
                        if not 0 <= t[0] < N or t in bars or t in cups:
                            break
                        wet.add(t)
                        if add(t, self.g) not in bars and add(t, self.g) not in cups:
                            streams.append(t)
                            break
                continue
            streams.append(n)
        return filled, floor, wet

    def _grid(self, wet=(), filled=()) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)

        def fill(c, colour):
            x, y = c
            g[y * C:(y + 1) * C, x * C:(x + 1) * C] = colour

        for c in self.floor:
            fill(c, FLOOR)
        for k, cs in enumerate(self.cup_cells):
            for c in cs:
                fill(c, FILLED if k in filled else CUP)
        for c in wet:
            fill(c, WATER)
        fill(self.spout, SPOUT)
        fill(add(self.spout, self.g), WATER)
        for i in range(len(self.bars)):
            for c in self._bar_cells(i):
                fill(c, SEL if i == self.selected else BAR)
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, 2, frames, [1, 2, 3, 4, 5, 6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        if action is GameAction.ACTION6:
            c = (data["x"] // C, data["y"] // C)
            hit = [i for i in range(len(self.bars)) if c in self._bar_cells(i)]
            if hit:
                self.selected = hit[0]
        elif action in KEYS:
            i = self.selected
            (x, y), w = self.bars[i]
            d = KEYS[action]
            at = (x + d[0], y + d[1])
            cells = self._bar_cells(i, at)
            if all(0 <= a < N and 0 <= b < N for a, b in cells) and not cells & self._solid():
                self.bars[i][0] = at
        elif action is GameAction.ACTION5:
            filled, floor, wet = self._flow()
            poured = self._grid(wet, filled)
            if len(filled) == len(self.cups) and not floor:
                self.done += 1
                if self.done == 2:
                    self.state = GameState.WIN
                    return self._obs([poured])
                self._load()
                return self._obs([poured, self._grid()])
            return self._obs([poured, self._grid()])
        return self._obs([self._grid()])


def test_pour_solver_fills_the_cups_and_keeps_the_floor_dry():
    env = PourRoom()
    agent = LibraryAgent(seed=0)
    obs = env.reset()
    done_at = []
    for used in range(1, 120):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(done_at) < obs.levels_completed:
            done_at.append(used)
        if obs.state == GameState.WIN:
            break
    solver = next(s for s in agent.solvers if s.name == "pour")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    assert per[0] <= 8, (per, dict(solver.stats))
    assert per[1] <= 10, (per, dict(solver.stats))
    assert solver.stats["pour_failed"] == 0, dict(solver.stats)
