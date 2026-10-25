"""The spell solver on a small game: rooms of floor in a wall, an avatar (a square of two
colours, top and bottom), an exit (a box in the avatar's colours) and a pad of 3x3 cells
that clicks toggle. When the cells toggled on make the level's spell it is cast and the pad
clears: on the first level the avatar shrinks to half its size and can pass a narrow
corridor, on the second it jumps to a marker in a room it cannot walk to, on the third a
shot leaves it the way it last moved and, if it meets the target, takes the target and the
barrier of the target's core colour away. The pad shows each level's spell in a colour of
its own. A move takes the avatar its size along (half that when a large one is blocked).
Colours, cell sizes, spell patterns and layout are none of sc25's, and the rules here are
written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

WALL, FLOOR, TOP, BOTTOM, MARK, FRAME, BLANK, HINT, ON, TARGET, CORE = 8, 1, 3, 12, 9, 4, 7, 13, 6, 10, 11
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
PAD_X, PAD_Y, CELL, PITCH = 24, 50, 2, 4

# each level: floor rectangles (x1, y1, x2, y2), the avatar (x, y, size), the exit box's
# top left (5x5), the spell (cells 0-8 by rows) and what it does
LEVELS = [
    dict(floor=[(20, 10, 47, 13), (7, 10, 19, 11)], avatar=(40, 10, 4), exit=(2, 8),
         spell={0, 4, 8}, kind="shrink"),
    dict(floor=[(24, 30, 43, 41), (24, 2, 43, 21)], avatar=(30, 34, 4), exit=(29, 2),
         spell={2, 4, 6}, kind="jump", mark=(30, 14)),
    dict(floor=[(20, 6, 45, 25), (46, 12, 51, 15), (26, 26, 29, 33)], avatar=(30, 12, 4), exit=(25, 34),
         spell={3, 4, 5}, kind="shot", target=(52, 12), barrier=(26, 26, 29, 29)),
]


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


class SpellRoom:
    def __init__(self) -> None:
        self.done = 0
        self.facing = (0, 1)
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        lv = LEVELS[self.done]
        self.lv = lv
        self.floor = np.zeros((64, 64), dtype=bool)
        for x1, y1, x2, y2 in lv["floor"]:
            self.floor[y1:y2 + 1, x1:x2 + 1] = True
        self.x, self.y, self.s = lv["avatar"]
        ex, ey = lv["exit"]
        self.exit = (ex, ey, ex + 4, ey + 4)
        self.target = lv.get("target")
        self.barrier = lv.get("barrier")
        if self.barrier is not None:
            x1, y1, x2, y2 = self.barrier
            self.floor[y1:y2 + 1, x1:x2 + 1] = False
        self.on: set = set()
        self.hint = True

    def _touches_exit(self, x, y, s) -> bool:
        x1, y1, x2, y2 = self.exit
        return x <= x2 and x + s > x1 and y <= y2 and y + s > y1

    def _open(self, x, y, s) -> bool:
        if x < 0 or y < 0 or x + s > 64 or y + s > 64:
            return False
        return bool(self.floor[y:y + s, x:x + s].all())

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), WALL, dtype=np.int8)
        g[self.floor] = FLOOR
        if self.barrier is not None:
            x1, y1, x2, y2 = self.barrier
            g[y1:y2 + 1, x1:x2 + 1] = CORE
        if self.target is not None:
            tx, ty = self.target
            g[ty:ty + 4, tx:tx + 4] = TARGET
            g[ty + 1:ty + 3, tx + 1:tx + 3] = CORE
        if "mark" in self.lv:
            mx, my = self.lv["mark"]
            for px, py in ((mx - 1, my - 1), (mx + 4, my - 1), (mx - 1, my + 4), (mx + 4, my + 4)):
                g[py, px] = MARK
        x1, y1, x2, y2 = self.exit
        g[y1:y2 + 1, x1:x2 + 1] = TOP
        g[y1 + 1:y2, x1 + 1:x2] = BOTTOM
        h = self.s // 2
        g[self.y:self.y + h, self.x:self.x + self.s] = TOP
        g[self.y + h:self.y + self.s, self.x:self.x + self.s] = BOTTOM
        g[PAD_Y - 2:PAD_Y + 3 * PITCH, PAD_X - 2:PAD_X + 3 * PITCH] = FRAME
        for i in range(9):
            cx, cy = PAD_X + (i % 3) * PITCH, PAD_Y + (i // 3) * PITCH
            colour = ON if i in self.on else HINT if self.hint and i in self.lv["spell"] else BLANK
            g[cy:cy + CELL, cx:cx + CELL] = colour
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(LEVELS), frames, [1, 2, 3, 4, 6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def _cast(self) -> None:
        kind = self.lv["kind"]
        if kind == "shrink":
            self.s = 2
        elif kind == "jump":
            self.x, self.y = self.lv["mark"]
        elif kind == "shot" and self.target is not None:
            dx, dy = self.facing
            lanes = [(self.x + (self.s - 1 if dx > 0 else 0), self.y + i) for i in range(self.s)] if dx else \
                    [(self.x + i, self.y + (self.s - 1 if dy > 0 else 0)) for i in range(self.s)]
            tx, ty = self.target
            for px, py in lanes:
                while True:
                    px, py = px + dx, py + dy
                    if not (0 <= px < 64 and 0 <= py < 64):
                        break
                    if tx <= px < tx + 4 and ty <= py < ty + 4:
                        x1, y1, x2, y2 = self.barrier
                        self.floor[y1:y2 + 1, x1:x2 + 1] = True
                        self.floor[ty:ty + 4, tx:tx + 4] = True
                        self.target = self.barrier = None
                        return
                    if not self.floor[py, px]:
                        break

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        if action is GameAction.ACTION6:
            i = None
            for k in range(9):
                cx, cy = PAD_X + (k % 3) * PITCH, PAD_Y + (k // 3) * PITCH
                if cx <= data["x"] < cx + CELL and cy <= data["y"] < cy + CELL:
                    i = k
            if i is not None:
                self.on ^= {i}
                if self.on == self.lv["spell"]:
                    self._cast()
                    self.on = set()
                    self.hint = False
            return self._obs([self._grid()])
        if action in KEYS:
            d = KEYS[action]
            self.facing = d
            for n in (self.s, self.s // 2) if self.s >= 4 else (self.s,):
                nx, ny = self.x + d[0] * n, self.y + d[1] * n
                if self._touches_exit(nx, ny, self.s):
                    finished = self._grid()
                    self.done += 1
                    if self.done == len(LEVELS):
                        self.state = GameState.WIN
                        return self._obs([finished])
                    self._load()
                    return self._obs([finished, self._grid()])
                if self._open(nx, ny, self.s):
                    self.x, self.y = nx, ny
                    break
        return self._obs([self._grid()])


def test_spell_solver_learns_each_spell_and_walks_to_the_exit():
    env = SpellRoom()
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
    solver = next(s for s in agent.solvers if s.name == "spells")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    # each spell is cast once to see what it does: the shrink before the walk (15 would walk
    # first), the jump, and a shot at the wall above before the one aimed at the target
    assert per[0] <= 20, (per, dict(solver.stats))
    assert per[1] <= 5, (per, dict(solver.stats))
    assert per[2] <= 14, (per, dict(solver.stats))
    for kind in ("resize", "jump", "shot"):
        assert solver.stats[f"learned_{kind}"] == 1, dict(solver.stats)
    assert solver.stats["cast_no_effect"] == 1, dict(solver.stats)
    # here a full step that touches the exit ends the level, which the solver learns
    assert solver.stats["learned_exit_full"] == 1, dict(solver.stats)
