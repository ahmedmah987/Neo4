"""The gravity solver on a small game: a shaft of 5-pixel cells taller than the screen, the view
following the avatar (a square in two colours of its own). Two keys move it a cell left or
right, and it falls the way gravity pulls (down at first) until a solid cell stops it. A click
clears a block, opens or shuts a gate, clears a turner and turns gravity round, or clears a
grower, which grows into every empty cell beside it but the avatar's. A fall onto
spikes, or running out of the bar of steps along the bottom row, ends the go. Reaching the gem
ends the level. Colours, sizes, the drawing, which way gravity first pulls and the layouts are
none of bp35's, and the rules here are written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, WALL, DOT, BLOCK, GATE, TURNER, SPIKE, GEM, GEM2, AV, AV2, BAR, GROW = 2, 8, 6, 12, 3, 14, 13, 4, 11, 9, 15, 7, 5
P, X0 = 5, 2  # the cell's size; the first column's pixel
BUDGET = 60  # actions in one go at a level
SOLID = set("#xTg^+y")

# '#' wall, '.' air, 'x' block, 'T' gate shut, 't' gate open, 'g' turner, '^' spikes, '+' gem,
# 'y' grower, '@' start
LEVELS = [
    # blocks in the way down, the gem far below the first view
    ["############",
     "#@.........#",
     "#####xx#####",
     "#..........#",
     "#..........#",
     "##xx########",
     "#..........#",
     "#..........#",
     "########xx##",
     "#..........#",
     "#..........#",
     "###xx#######",
     "#..........#",
     "#......+...#",
     "############"],
    # a cheap way down drops onto spikes out of sight; the other way is longer
    ["############",
     "#@.........#",
     "##xx####...#",
     "#......#...#",
     "#......#...#",
     "#......#...#",
     "#......#...#",
     "#......#...#",
     "#......#...#",
     "#......#xx##",
     "#^^^^^^#...#",
     "########...#",
     "#..........#",
     "#.........+#",
     "############"],
    # the only way down falls through an open gate onto spikes, unless the gate is shut first
    ["############",
     "#..@.......#",
     "#.##########",
     "#......+...#",
     "#t##########",
     "#.#........#",
     "#^#........#",
     "############"],
    # the gem is above: a turner sends gravity the other way
    ["############",
     "#.....+....#",
     "#.######...#",
     "#..........#",
     "#..........#",
     "#..........#",
     "#.@.....g..#",
     "############"],
    # growers: dug down through, and tunnelled through sideways
    ["############",
     "#..@......##",
     "###y########",
     "###.########",
     "###.########",
     "###....y..+#",
     "############"],
    # the way to the gem crosses a hole above spikes: a chain of grower clicks plugs it
    ["############",
     "#@....+....#",
     "###.########",
     "#y..########",
     "#^^^^#######",
     "############"],
    # turners shut in a wall, all alike: gravity is turned up and down again to reach the gem
    ["############",
     "#g#........#",
     "#g#.######.#",
     "#g#.#....#.#",
     "#g#.#.##.#.#",
     "#g#...#+.#.#",
     "#g#@..####.#",
     "############"],
]


def tile(ch: str) -> np.ndarray:
    t = np.full((P, P), BG, dtype=np.int8)
    if ch == "#":
        t[:] = WALL
        t[1, 1] = t[3, 3] = DOT
    elif ch == "x":
        t[:] = BLOCK
        t[0, 0] = t[0, P - 1] = t[P - 1, 0] = t[P - 1, P - 1] = WALL
    elif ch == "T":
        t[:] = GATE
    elif ch == "t":
        t[::2, ::2] = GATE
        t[1::2, 1::2] = GATE
    elif ch == "g":
        t[:] = TURNER
        t[2, 2] = 0
    elif ch == "^":
        t[:] = SPIKE
        t[0, ::2] = BG
    elif ch == "+":
        t[:] = GEM2
        t[1:4, 2] = t[2, 1:4] = GEM
    elif ch == "y":
        t[:] = GROW
        t[0, 2] = t[2, 0] = t[2, 4] = t[4, 2] = BG
    return t


def obs_of(state, done, frames) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=len(LEVELS), available_actions=[3, 4, 6])
    o.frame = frames
    return o


class ShaftRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_FINISHED

    def _load(self) -> None:
        rows = LEVELS[self.done]
        self.map = [list(r) for r in rows]
        for y, r in enumerate(self.map):
            for x, ch in enumerate(r):
                if ch == "@":
                    self.at = (x, y)
                    r[x] = "."
        self.g = 1
        self.left = BUDGET
        self.cam = self._cam_target()

    def _cam_target(self) -> int:
        """The screen row of the avatar's cell's top: high when it falls down, low when up."""
        return 20 if self.g > 0 else 40

    def _cell(self, c) -> str:
        x, y = c
        if 0 <= y < len(self.map) and 0 <= x < len(self.map[0]):
            return self.map[y][x]
        return "#"

    def _grid(self, at=None, cam=None) -> np.ndarray:
        at = self.at if at is None else at
        cam = self.cam if cam is None else cam
        g = np.full((64, 64), BG, dtype=np.int8)
        g[:, :X0] = WALL
        top_row = at[1] - (cam // P) - 1
        for cy in range(top_row, top_row + 64 // P + 3):
            y = cam + (cy - at[1]) * P
            for cx in range(len(self.map[0]) + 1):
                x = X0 + cx * P
                t = tile(self._cell((cx, cy)))
                y1, y2 = max(0, y), min(63, y + P)
                x1, x2 = max(0, x), min(64, x + P)
                if y1 < y2 and x1 < x2:
                    g[y1:y2, x1:x2] = t[y1 - y:y2 - y, x1 - x:x2 - x]
        ay, ax = cam, X0 + at[0] * P
        g[ay:ay + 4, ax + 1:ax + 5] = AV
        g[ay + 1:ay + 3, ax + 2:ax + 4] = AV2
        g[63] = 0
        g[63, :BUDGET - self.left] = BAR
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, frames)

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def _fall(self, frames: list) -> bool:
        """Fall the way gravity pulls, a frame a cell; False on spikes."""
        while True:
            n = (self.at[0], self.at[1] + self.g)
            ch = self._cell(n)
            if ch == "+":
                return True
            if ch in SOLID:
                return ch != "^"
            self.at = n
            frames.append(self._grid())

    def _end(self, frames: list) -> FrameDataRaw:
        """The level done if the avatar is on the gem or rests on it, the go over if it died."""
        n = (self.at[0], self.at[1] + self.g)
        if self._cell(self.at) == "+" or self._cell(n) == "+":
            self.done += 1
            if self.done == len(LEVELS):
                self.state = GameState.WIN
                return self._obs(frames)
            self._load()
            return self._obs(frames + [self._grid()])
        if self.left <= 0:
            self.state = GameState.GAME_OVER  # the bar is full
            return self._obs(frames or [self._grid()])
        return self._obs(frames or [self._grid()])

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        self.left -= 1
        frames: list = []
        if action in (GameAction.ACTION3, GameAction.ACTION4):
            t = (self.at[0] + (1 if action is GameAction.ACTION4 else -1), self.at[1])
            ch = self._cell(t)
            if ch == "+":
                self.at = t
                return self._end([self._grid()])
            if ch in SOLID:
                return self._end([self._grid()])
            self.at = t
            frames.append(self._grid())
            if not self._fall(frames):
                self.state = GameState.GAME_OVER  # on the spikes
                return self._obs(frames)
            return self._end(frames)
        if action is GameAction.ACTION6:
            x = (data["x"] - X0) // P
            y = self.at[1] + (data["y"] - self.cam) // P
            ch = self._cell((x, y))
            if ch == "y" and 0 < y < len(self.map) - 1:
                self.map[y][x] = "."
                for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if n != self.at and self._cell(n) == ".":
                        self.map[n[1]][n[0]] = "y"
                frames.append(self._grid())
                if not self._fall(frames):
                    self.state = GameState.GAME_OVER  # on the spikes
                    return self._obs(frames)
            elif ch in "xTtg" and 0 < y < len(self.map) - 1:
                self.map[y][x] = {"x": ".", "T": "t", "t": "T", "g": "."}[ch]
                if ch == "g":
                    self.g = -self.g
                    want = self._cam_target()
                    while self.cam != want:  # the view slides to the avatar's other place
                        self.cam += P if want > self.cam else -P
                        frames.append(self._grid())
                frames.append(self._grid())
                if not self._fall(frames):
                    self.state = GameState.GAME_OVER  # on the spikes
                    return self._obs(frames)
            return self._end(frames)
        return self._end(frames)


def test_gravity_solver_explores_the_shaft_and_reaches_the_gem():
    env = ShaftRoom()
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
    solver = next(s for s in agent.solvers if s.name == "gravity")
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    # level 1: a first move shows the avatar and the step; a block clicked once shows it clears;
    # the way down the shaft is found a view at a time, until the gem comes into sight
    assert per[0] <= 22, (per, dict(solver.stats))
    # level 2: the cheapest way on is a fall into rows not yet seen, onto spikes; the level is
    # begun again with the map kept, and the other way taken
    assert per[1] <= 13, (per, dict(solver.stats))
    # level 3: the open gate is clicked to see what it does (it shuts), and the fall stops on it
    assert per[2] <= 9, (per, dict(solver.stats))
    # level 4: the turner is clicked to see what it does: gravity turns, and the gem is above
    assert per[3] <= 7, (per, dict(solver.stats))
    # level 5: a grower clicked to see what it does grows into the empty cells beside it; the
    # avatar digs down through growers (each regrows below it) and tunnels through one sideways
    assert per[4] <= 18, (per, dict(solver.stats))
    # level 6: a chain of three grower clicks grows a grower into the hole above the spikes
    assert per[5] <= 8, (per, dict(solver.stats))
    # level 7: two of six alike turners shut in a wall turn gravity up and down again
    assert per[6] <= 8, (per, dict(solver.stats))
    assert solver.stats["learned_grow"] == 1, dict(solver.stats)
    assert solver.stats["probe"] == 1 and solver.stats["learned_goal"] == 1, dict(solver.stats)
    assert solver.stats["learned_deadly"] == 1, dict(solver.stats)
    assert solver.stats["learned_become"] == 2 and solver.stats["learned_gravity"] == 1, dict(solver.stats)
