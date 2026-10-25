"""The spell solver on a small game: rooms of floor in a wall, an avatar (a square of two
colours, top and bottom), an exit (a box in the avatar's colours) and a pad of 3x3 cells
that clicks toggle. When the cells toggled on make one of the level's spells it is cast and
the pad clears: the avatar shrinks to half its size (and can pass a narrow corridor), or it
jumps to the next mark (marks are visited in turn; a ring is drawn round the next), or a
shot leaves it the way it last moved, flies over floor and posts, and if it meets a target
takes the target and the barrier of the target's core colour away. The first three levels
show their spell on the pad in a colour of their own; the fourth (a shrink and a shot down a
narrow corridor past posts, at a target of another core colour) and the fifth (two marks,
the first a dead end) show none. A move takes the avatar its size along (half that when a
large one is blocked); posts stop it. Colours, cell sizes, spell patterns and layout are
none of sc25's, and the rules here are written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

WALL, FLOOR, TOP, BOTTOM, MARK, HALO, FRAME, BLANK, HINT, ON, TARGET, CORE, CORE2, POST = \
    8, 1, 3, 12, 9, 15, 4, 7, 13, 6, 10, 11, 2, 5
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
PAD_X, PAD_Y, CELL, PITCH = 24, 50, 2, 4
SHRINK, JUMP, SHOT = frozenset({0, 4, 8}), frozenset({2, 4, 6}), frozenset({3, 4, 5})

# each level: floor rectangles (x1, y1, x2, y2), posts (cells), the avatar (x, y, size), the
# exit box's top left (5x5), the spells, the one the pad shows, marks, a target (top left of
# a 4x4 box) with its core colour and the barrier of that colour (x1, y1, x2, y2)
LEVELS = [
    dict(floor=[(20, 10, 47, 13), (7, 10, 19, 11)], avatar=(40, 10, 4), exit=(2, 8),
         spells={SHRINK: "shrink"}, hint=SHRINK),
    dict(floor=[(24, 30, 43, 41), (24, 2, 43, 21)], avatar=(30, 34, 4), exit=(29, 2),
         spells={JUMP: "jump"}, hint=JUMP, marks=[(30, 14)]),
    dict(floor=[(20, 6, 45, 25), (46, 12, 51, 15), (26, 26, 29, 33)], posts=[(48, 12), (49, 12), (48, 15), (49, 15)],
         avatar=(30, 12, 4), exit=(25, 34), spells={SHOT: "shot"}, hint=SHOT,
         target=(52, 12), core=CORE, barrier=(26, 26, 29, 29)),
    dict(floor=[(20, 6, 45, 25), (8, 16, 19, 17), (46, 20, 49, 23)], posts=[(12, 16), (13, 16), (12, 17), (13, 17)],
         avatar=(24, 8, 4), exit=(50, 20), spells={SHRINK: "shrink", SHOT: "shot"},
         target=(4, 15), core=CORE2, barrier=(46, 20, 49, 23)),
    dict(floor=[(4, 4, 19, 19), (40, 4, 59, 19), (4, 30, 43, 41)], avatar=(8, 8, 4), exit=(36, 33),
         spells={JUMP: "jump"}, marks=[(48, 10), (10, 34)]),
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
        self.posts = set(lv.get("posts", []))
        for px, py in self.posts:
            self.floor[py, px] = False
        self.x, self.y, self.s = lv["avatar"]
        ex, ey = lv["exit"]
        self.exit = (ex, ey, ex + 4, ey + 4)
        self.target = lv.get("target")
        self.barrier = lv.get("barrier")
        if self.barrier is not None:
            x1, y1, x2, y2 = self.barrier
            self.floor[y1:y2 + 1, x1:x2 + 1] = False
        self.marks = lv.get("marks", [])
        self.next_mark = 0
        self.on: set = set()
        self.hint = lv.get("hint")

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
        for px, py in self.posts:
            g[py, px] = POST
        if self.barrier is not None:
            x1, y1, x2, y2 = self.barrier
            g[y1:y2 + 1, x1:x2 + 1] = self.lv["core"]
        if self.target is not None:
            tx, ty = self.target
            g[ty:ty + 4, tx:tx + 4] = TARGET
            g[ty + 1:ty + 3, tx + 1:tx + 3] = self.lv["core"]
        for i, (mx, my) in enumerate(self.marks):
            for px, py in ((mx, my), (mx + 3, my), (mx, my + 3), (mx + 3, my + 3)):
                g[py, px] = MARK
            if i == self.next_mark:
                for px, py in ((mx - 1, my - 1), (mx + 4, my - 1), (mx - 1, my + 4), (mx + 4, my + 4)):
                    g[py, px] = HALO
        x1, y1, x2, y2 = self.exit
        g[y1:y2 + 1, x1:x2 + 1] = TOP
        g[y1 + 1:y2, x1 + 1:x2] = BOTTOM
        h = self.s // 2
        g[self.y:self.y + h, self.x:self.x + self.s] = TOP
        g[self.y + h:self.y + self.s, self.x:self.x + self.s] = BOTTOM
        g[PAD_Y - 2:PAD_Y + 3 * PITCH, PAD_X - 2:PAD_X + 3 * PITCH] = FRAME
        for i in range(9):
            cx, cy = PAD_X + (i % 3) * PITCH, PAD_Y + (i // 3) * PITCH
            colour = ON if i in self.on else HINT if self.hint and i in self.hint else BLANK
            g[cy:cy + CELL, cx:cx + CELL] = colour
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(LEVELS), frames, [1, 2, 3, 4, 6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def _shot(self) -> None:
        if self.target is None:
            return
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
                if not self.floor[py, px] and (px, py) not in self.posts:
                    break

    def _cast(self, kind) -> None:
        if kind == "shrink":
            self.s = 2
        elif kind == "jump" and self.marks:
            self.x, self.y = self.marks[self.next_mark]
            self.next_mark = (self.next_mark + 1) % len(self.marks)
        elif kind == "shot":
            self._shot()

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
                kind = self.lv["spells"].get(frozenset(self.on))
                if kind is not None:
                    self._cast(kind)
                    self.on = set()
                    self.hint = None
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
    for used in range(1, 300):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(done_at) < obs.levels_completed:
            done_at.append(used)
        if obs.state == GameState.WIN:
            break
    solver = next(s for s in agent.solvers if s.name == "spells")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    # each shown spell is cast once to see what it does: the shrink before the walk (15
    # would walk first), the jump, and a shot at the wall above before the one aimed at the
    # target, past the posts it learns a shot flies through
    assert all(a <= b for a, b in zip(per[:3], [20, 5, 14])), (per, dict(solver.stats))
    # from memory: a shrink, a shot down the narrow corridor past the posts (and one try at
    # growing back, which this game does not do); two jumps in turn to reach the exit's room
    assert per[3] <= 24, (per, dict(solver.stats))
    assert per[4] <= 12, (per, dict(solver.stats))
    for kind in ("resize", "shot", "target", "shot_through"):
        assert solver.stats[f"learned_{kind}"] >= 1, dict(solver.stats)
    assert solver.stats["learned_jump"] == 3, dict(solver.stats)
    assert solver.stats["cast_no_effect"] == 1, dict(solver.stats)
    assert solver.stats["unlearned_resize"] == 1, dict(solver.stats)
    # here a full step that touches the exit ends the level, which the solver learns
    assert solver.stats["learned_exit_full"] == 1, dict(solver.stats)
