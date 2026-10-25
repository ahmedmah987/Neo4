"""The key solver on a small game: a maze of 6-pixel cells, an avatar drawn as two bands,
a key (a 3x3 glyph in one colour) shown at twice the size in a box, doors that take the
avatar only when the key is their glyph, tiles that turn the key a quarter, give it the
next colour or the next shape, a bar of steps left with a refill, a wall whose edge line
throws the avatar from the cell in front of it to the next wall, and on the third level a
colour tile that slides up and down a corridor. The fourth level lies in fog: only what is
within 14 pixels of the avatar is drawn (the key and the bar are drawn at the bottom, over
the fog), and the door is out of sight at the start. When the steps run out the level starts
over. Colours, cell size, shapes and layout are none of ls20's, and the rules here are
written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

WALL, FLOOR, BOX, TOP, BOTTOM, BAR, LINE, FOG = 2, 7, 1, 4, 12, 11, 15, 8
KEY_COLOURS = (6, 10, 13)
SHAPES = (
    ((1, 0, 0), (1, 0, 0), (1, 1, 0)),
    ((1, 1, 1), (0, 1, 0), (0, 0, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 0, 1)),
)
ICONS = {
    "turn": ((3, 0, 3), (0, 3, 0), (3, 0, 3)),
    "colour": ((FLOOR, 9, FLOOR), (9, 9, 9), (FLOOR, 9, FLOOR)),
    "shape": ((14, 14, FLOOR), (14, FLOOR, FLOOR), (FLOOR, FLOOR, 5)),
    "refill": ((FLOOR, FLOOR, FLOOR), (FLOOR, BAR, FLOOR), (FLOOR, FLOOR, FLOOR)),
}
S, OFF, ROW = 6, 1, 60
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
TILES = {"T": "turn", "C": "colour", "S": "shape", "R": "refill"}

# each level: the map, the key at the start and each door's key as (shape, turns, colour),
# the steps a move costs, the bar, the edge lines (wall cell -> the side the line is on),
# the moving tiles (kind, the cells it slides along), and the fog's radius if any
LEVELS = [
    dict(rows=["#########",
               "#A..T...#",
               "#.#####.#",
               "#.......#",
               "#.#.###.#",
               "#C#.....#",
               "#.#####.#",
               "#......1#",
               "#########"],
         key=(0, 0, 0), doors={"1": (0, 1, 1)}, cost=1, bar=30, lines={}, movers=[]),
    dict(rows=["#########",
               "#A......#",
               "#.#####.#",
               "#.#1.S..#",
               "#.#####.#",
               "#.#R....#",
               "#.#######",
               "#2#######",
               "#########"],
         key=(0, 0, 0), doors={"1": (1, 0, 0), "2": (2, 0, 0)}, cost=2, bar=44,
         lines={(1, 0): (0, 1)}, movers=[]),
    dict(rows=["#########",
               "#A......#",
               "#######.#",
               "#1......#",
               "#########",
               "#########",
               "#########",
               "#########",
               "#########"],
         key=(2, 0, 0), doors={"1": (2, 0, 2)}, cost=1, bar=40, lines={},
         movers=[("colour", [(3, 1), (4, 1), (5, 1)])]),
    dict(rows=["#########",
               "#A......#",
               "#.#####.#",
               "#.#...#.#",
               "#.#.#.#.#",
               "#...#C..#",
               "#####.###",
               "#1....###",
               "#########"],
         key=(1, 0, 0), doors={"1": (1, 0, 1)}, cost=1, bar=40, lines={}, movers=[], fog=14),
]


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def glyph(key):
    """The key's 3x3 glyph: its shape turned clockwise, in its colour on the box colour."""
    shape, turns, colour = key
    m = np.rot90(np.array(SHAPES[shape]), -turns)
    return np.where(m == 1, KEY_COLOURS[colour], BOX)


class KeyRoom:
    def __init__(self, levels=LEVELS) -> None:
        self.levels = levels
        self.done = 0
        self.lives = 3
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        lv = self.levels[self.done]
        self.walls, self.tiles, self.doors = set(), {}, {}
        for y, row in enumerate(lv["rows"]):
            for x, ch in enumerate(row):
                c = (x, y)
                if ch == "#":
                    self.walls.add(c)
                elif ch == "A":
                    self.avatar = c
                elif ch in TILES:
                    self.tiles[c] = TILES[ch]
                elif ch.isdigit():
                    self.doors[c] = lv["doors"][ch]
        self.key = lv["key"]
        self.cost, self.full = lv["cost"], lv["bar"]
        self.steps = self.full
        self.lines = lv["lines"]
        # pushers: the floor cell an edge line faces throws the avatar the same way
        self.pushers = {add(w, d): d for w, d in self.lines.items()}
        self.movers = [[kind, track, 0, 1] for kind, track in lv["movers"]]

    def _mover_cells(self):
        return {m[1][m[2]]: m[0] for m in self.movers}

    def _blocked(self, c) -> bool:
        return c in self.walls or not (0 <= c[0] < 9 and 0 <= c[1] < 9)

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), WALL, dtype=np.int8)

        def at(c):
            return OFF + c[1] * S, OFF + c[0] * S

        def icon(c, pattern):
            y, x = at(c)
            g[y + 2:y + 5, x + 2:x + 5] = np.array(pattern)

        for y in range(9):
            for x in range(9):
                c = (x, y)
                if c in self.walls:
                    continue
                y0, x0 = at(c)
                g[y0:y0 + S, x0:x0 + S] = FLOOR
                if c in self.doors:
                    g[y0:y0 + S, x0:x0 + S] = BOX
                    g[y0 + 2:y0 + 5, x0 + 2:x0 + 5] = glyph(self.doors[c])
        for w, d in self.lines.items():
            y0, x0 = at(w)
            if d == (0, 1):
                g[y0 + S - 1, x0:x0 + S] = LINE
            elif d == (0, -1):
                g[y0, x0:x0 + S] = LINE
            elif d == (1, 0):
                g[y0:y0 + S, x0 + S - 1] = LINE
            else:
                g[y0:y0 + S, x0] = LINE
        for c, kind in list(self.tiles.items()) + list(self._mover_cells().items()):
            icon(c, ICONS[kind])
        y0, x0 = at(self.avatar)
        g[y0:y0 + 2, x0:x0 + S] = TOP
        g[y0 + 2:y0 + S, x0:x0 + S] = BOTTOM
        fog = self.levels[self.done].get("fog")
        if fog:
            # all but a disc round the avatar is fog; the key and the bar are drawn over it
            ys, xs = np.mgrid[0:64, 0:64]
            g[np.hypot(xs - (x0 + (S - 1) / 2), ys - (y0 + (S - 1) / 2)) > fog] = FOG
            g[55:63, 1:9] = BOX
            g[56:62, 2:8] = np.repeat(np.repeat(glyph(self.key), 2, 0), 2, 1)
            g[63, 10:10 + self.full] = WALL
            g[63, 10:10 + self.steps] = BAR
            return g
        # the key box, top right, and the bar of steps left along the bottom
        g[1:9, 56:64] = BOX
        g[2:8, 57:63] = np.repeat(np.repeat(glyph(self.key), 2, 0), 2, 1)
        g[ROW, 2:2 + self.full] = WALL
        g[ROW, 2:2 + self.steps] = BAR
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(self.levels), frames, [1, 2, 3, 4])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def _effect(self, kind) -> None:
        shape, turns, colour = self.key
        if kind == "turn":
            self.key = (shape, (turns + 1) % 4, colour)
        elif kind == "colour":
            self.key = (shape, turns, (colour + 1) % len(KEY_COLOURS))
        elif kind == "shape":
            self.key = ((shape + 1) % len(SHAPES), turns, colour)

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action not in KEYS:
            return self._obs([self._grid()])
        d = KEYS[action]
        t = add(self.avatar, d)
        if self._blocked(t) or (t in self.doors and self.doors[t] != self.key):
            return self._obs([self._grid()])
        self.doors.pop(t, None)
        if t in self.pushers:
            p = self.pushers[t]
            while not self._blocked(add(t, p)) and add(t, p) not in self.doors:
                t = add(t, p)
        for m in self.movers:
            kind, track, i, way = m
            if not 0 <= i + way < len(track):
                way = -way
            m[2], m[3] = i + way, way
        here = self.tiles.get(t) or self._mover_cells().get(t)
        if here == "refill":
            self.steps = self.full
            del self.tiles[t]
        elif self.steps < self.cost:
            # out of steps: a life is lost and the level starts over
            self.lives -= 1
            if self.lives == 0:
                self.state = GameState.GAME_OVER
                return self._obs([])
            self._load()
            return self._obs([self._grid()])
        else:
            self.steps -= self.cost
            if here is not None:
                self._effect(here)
        self.avatar = t
        if not self.doors:
            finished = self._grid()
            self.done += 1
            if self.done == len(self.levels):
                self.state = GameState.WIN
                return self._obs([finished])
            self._load()
            return self._obs([finished, self._grid()])
        return self._obs([self._grid()])


def test_key_solver_turns_recolours_and_reshapes_the_key_into_every_door():
    env = KeyRoom()
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
    solver = next(s for s in agent.solvers if s.name == "keys")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    # the fewest moves are 18, 37 and 16; the third level's tile is watched before it is used
    assert per[0] <= 18, (per, dict(solver.stats))
    assert per[1] <= 37, (per, dict(solver.stats))
    assert per[2] <= 22, (per, dict(solver.stats))
    # in fog: down the left side and round to the colour tile, its colour known, then the door
    # once it is seen; 18 is also the fewest with the whole map in view
    assert per[3] <= 18, (per, dict(solver.stats))
    assert solver.stats["fog_levels"] == 1 and solver.stats["fog_walks"] >= 1, dict(solver.stats)
    assert solver.stats["lives_lost"] == 0, dict(solver.stats)
    for kind in ("turn", "colour", "shape", "push"):
        assert solver.stats[f"learned_{kind}"] >= 1, dict(solver.stats)
    assert solver.stats["cycles_learned"] == 1, dict(solver.stats)
