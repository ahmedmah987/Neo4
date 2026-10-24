"""The bit-program solver on a small game: a board with a piece and its outline, a
panel of bit columns under it and a button; each column's value is an instruction
(not the sum of its bits: 1 moves right, 2 down, 3 up, 4 turns, ...) run one column a
frame; a piece that ends on its outline completes the level, any other goes back to
its start, or to a mark it ended on. The second level has a demo board whose tabs run
programs that show what other values do; the third needs two runs through a mark.
Colours, sizes and the piece's shape are none of tn36's. Nothing in the solver knows
which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, FLOOR, DEMO_FLOOR, GROUND, DEMO_GROUND = 0, 6, 3, 2, 1
ON, OFF, PIECE, DEMO_PIECE, BUTTON, TAB = 7, 8, 9, 10, 12, 11
BASE = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool)
STEP = 3
TABLE = {1: (STEP, 0, 0), 2: (0, STEP, 0), 3: (0, -STEP, 0), 4: (0, 0, 1), 5: (-STEP, 0, 0)}
BOARD = (34, 4, 60, 25)  # x1, y1, x2, y2
DEMO_BOARD = (2, 4, 28, 25)


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def _mask(k: int) -> np.ndarray:
    return np.rot90(BASE, k)


class BitRoom:
    # (start (x, y, k), target (x, y, k), starting program, demos, mark)
    LEVELS = [
        ((37, 7, 0), (49, 7, 0), [1, 0, 1, 0], [], None),
        ((40, 16, 0), (40, 10, 1), [0, 0, 0, 0], [[4, 4, 4], [3, 3, 3]], None),
        ((37, 13, 0), (55, 13, 0), [0, 0, 0], [], (46, 13)),
    ]

    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        self.start, self.target, prog, self.demos, self.mark = self.LEVELS[self.done]
        self.piece = self.start
        self.bits = [[bool(v >> i & 1) for i in range(3)] for v in prog]
        self.demo_bits = [[bool(v >> i & 1) for i in range(3)] for v in (self.demos[0] if self.demos else [])]
        self.demo_piece = (12, 16, 0)

    # --- drawing ---------------------------------------------------------------------

    @staticmethod
    def _glyphs(g, cols_x, bits, ground_y) -> None:
        for x, col in zip(cols_x, bits):
            for i, on in enumerate(col):
                y = ground_y + 2 + 3 * i
                c = ON if on else OFF
                if i == 0:
                    g[y, x:x + 2] = c
                elif i == 1:
                    g[y:y + 2, x] = c
                else:
                    g[y, x] = c

    def _cols_x(self):
        return [37 + 6 * i for i in range(len(self.bits))]

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)
        x1, y1, x2, y2 = BOARD
        g[y1:y2 + 1, x1:x2 + 1] = FLOOR
        g[26:37, 34:61] = GROUND
        self._glyphs(g, self._cols_x(), self.bits, 26)
        g[40:45, 45:50] = BUTTON
        if self.mark is not None:
            mx, my = self.mark
            g[my:my + 3, mx:mx + 3] = PIECE
        tx, ty, tk = self.target
        grown = np.zeros((5, 5), dtype=bool)
        m = _mask(tk)
        for dy in range(3):
            for dx in range(3):
                grown[dy:dy + 3, dx:dx + 3] |= m
        grown[1:4, 1:4] &= ~m
        g[ty - 1:ty + 4, tx - 1:tx + 4][grown] = PIECE
        px, py, pk = self.piece
        g[py:py + 3, px:px + 3][_mask(pk)] = PIECE
        if self.demos:
            dx1, dy1, dx2, dy2 = DEMO_BOARD
            g[dy1:dy2 + 1, dx1:dx2 + 1] = DEMO_FLOOR
            g[26:37, 2:29] = DEMO_GROUND
            self._glyphs(g, [6 + 6 * i for i in range(len(self.demo_bits))], self.demo_bits, 26)
            px, py, pk = self.demo_piece
            g[py:py + 3, px:px + 3][_mask(pk)] = DEMO_PIECE
            for i in range(len(self.demos)):
                x = 3 + 9 * i
                g[40:47, x:x + 7] = TAB
                g[41:46, x + 1:x + 6] = BG
                g[43, x + 3] = TAB
        return g

    # --- playing ---------------------------------------------------------------------

    @staticmethod
    def _apply(piece, value, board):
        dx, dy, dk = TABLE.get(value, (0, 0, 0))
        x, y, k = piece
        nx, ny = x + dx, y + dy
        if nx < board[0] or ny < board[1] or nx + 2 > board[2] or ny + 2 > board[3]:
            return piece
        return (nx, ny, (k + dk) % 4)

    @staticmethod
    def _value(col) -> int:
        return sum(1 << i for i, on in enumerate(col) if on)

    def _run(self) -> list:
        frames = []
        self.piece = self.start
        frames.append(self._grid())
        for col in self.bits:
            self.piece = self._apply(self.piece, self._value(col), BOARD)
            frames.append(self._grid())
        if self.piece == self.target:
            return frames
        if self.mark is not None and self.piece[:2] == self.mark:
            self.start = self.piece
        self.piece = self.start
        frames.append(self._grid())
        return frames

    def _demo(self, i: int) -> list:
        prog = self.demos[i]
        self.demo_bits = [[bool(v >> j & 1) for j in range(3)] for v in prog]
        self.demo_piece = (12, 16, 0)
        frames = [self._grid()]
        for v in prog:
            self.demo_piece = self._apply(self.demo_piece, v, DEMO_BOARD)
            frames.append(self._grid())
        frames.append(self._grid())
        return frames

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(self.LEVELS), frames, [6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action is not GameAction.ACTION6:
            return self._obs([self._grid()])
        x, y = data["x"], data["y"]
        for ci, cx in enumerate(self._cols_x()):
            for i in range(3):
                gy = 28 + 3 * i
                hit = (i == 0 and y == gy and cx <= x <= cx + 1) or (i == 1 and x == cx and gy <= y <= gy + 1) \
                    or (i == 2 and x == cx and y == gy)
                if hit:
                    self.bits[ci][i] = not self.bits[ci][i]
                    return self._obs([self._grid()])
        if 45 <= x < 50 and 40 <= y < 45:
            frames = self._run()
            if self.piece == self.target:
                self.done += 1
                if self.done == len(self.LEVELS):
                    self.state = GameState.WIN
                    return self._obs(frames)
                self._load()
                return self._obs(frames + [self._grid()])
            return self._obs(frames)
        for i in range(len(self.demos)):
            if 3 + 9 * i <= x < 10 + 9 * i and 40 <= y < 47:
                return self._obs(self._demo(i))
        return self._obs([self._grid()])


def test_bit_solver_learns_the_instructions_and_programs_the_piece_home():
    env = BitRoom()
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
    solver = next(s for s in agent.solvers if s.name == "bits")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats), solver.effects)
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    # a run to see what the panel holds, two flips, a run
    assert per[0] <= 4, (per, dict(solver.stats))
    # both demos (a turn, a move up), five flips, a run
    assert per[1] <= 8, (per, dict(solver.stats))
    # three flips; a run onto the mark, a run from it
    assert per[2] <= 5, (per, dict(solver.stats))
