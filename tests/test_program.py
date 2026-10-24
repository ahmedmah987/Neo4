"""The program-family solver on a small game: a target row of colours, frames with a
row of slots, a palette of tiles. Clicking a tile and then an empty slot moves the
tile there; ACTION5 runs the program (the first frame's slots in order, a hollow tile
running the frame of its colour); the level ends when the run writes the target.
Nothing in the solver knows which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, STRIP, MARK, PITCH = 4, 5, 2, 6


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


class ProgramRoom:
    # (target, frames as (colour, x, y, slots), palette); a slot is None (empty),
    # ("tile", colour) or ("call", colour)
    LEVELS = [
        ([9, 14, 11], [(8, 16, 20, [None, None, None])],
         [("tile", 14), ("tile", 11), ("tile", 9), ("tile", 12)]),
        ([6, 9, 12, 9, 12, 15], [(8, 16, 16, [None, ("call", 14), None, None]), (14, 16, 32, [None, None])],
         [("tile", 9), ("tile", 15), ("call", 14), ("tile", 12), ("tile", 6), ("tile", 11)]),
    ]

    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        self.target, frames, palette = self.LEVELS[self.done]
        self.frames = [(col, x, y, list(slots)) for col, x, y, slots in frames]
        self.palette = list(palette)
        self.sel = None

    def _slot_at(self, fi: int, si: int) -> tuple[int, int]:
        _, x, y, _ = self.frames[fi]
        return x + 5 + si * PITCH, y + 5  # the slot's centre

    def _pal_at(self, i: int) -> tuple[int, int]:
        return 10 + i * 7, 58

    @staticmethod
    def _draw(g, cell, cx, cy) -> None:
        if cell is None:
            g[cy - 1:cy + 1, cx - 1:cx + 1] = MARK
            return
        kind, col = cell
        g[cy - 2:cy + 2, cx - 2:cx + 2] = col
        if kind == "call":
            g[cy - 1:cy + 1, cx - 1:cx + 1] = BG

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)
        g[0:8, 8:8 + 7 * len(self.target) + 1] = STRIP
        for i, col in enumerate(self.target):
            x = 9 + 7 * i
            g[1:7, x:x + 6] = col
            g[2:6, x + 1:x + 5] = STRIP
        for fi, (col, x, y, slots) in enumerate(self.frames):
            w = 4 + PITCH * len(slots)
            g[y, x:x + w] = col
            g[y + 9, x:x + w] = col
            g[y:y + 10, x] = col
            g[y:y + 10, x + w - 1] = col
            for si, cell in enumerate(slots):
                self._draw(g, cell, *self._slot_at(fi, si))
        g[52, :] = 3
        for i, cell in enumerate(self.palette):
            if cell is not None:
                self._draw(g, cell, *self._pal_at(i))
        return g

    def _run(self) -> bool:
        by_col = {f[0]: i for i, f in enumerate(self.frames)}
        stack, t = [[0, 0]], 0
        for _ in range(200):
            if t == len(self.target):
                return True
            fi, si = stack[-1]
            if si >= len(self.frames[fi][3]):
                if len(stack) == 1:
                    return False
                stack.pop()
                stack[-1][1] += 1
                continue
            cell = self.frames[fi][3][si]
            if cell is None:
                return False
            kind, col = cell
            if kind == "tile":
                if col != self.target[t]:
                    return False
                t += 1
                stack[-1][1] += 1
            else:
                stack.append([by_col[col], 0])
        return False

    def _obs(self, frames=None) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(self.LEVELS), (frames or []) + [self._grid()], [5, 6, 7])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs()
        if action is GameAction.ACTION5:
            if self._run():
                finished = self._grid()
                self.done += 1
                if self.done == len(self.LEVELS):
                    self.state = GameState.WIN
                else:
                    self._load()
                return self._obs([finished])
            return self._obs()
        if action is not GameAction.ACTION6 or data is None:
            return self._obs()
        x, y = data["x"], data["y"]
        for i, cell in enumerate(self.palette):
            px, py = self._pal_at(i)
            if cell is not None and abs(x - px) <= 2 and abs(y - py) <= 2:
                self.sel = i
                return self._obs()
        if self.sel is not None:
            for fi, (_, _, _, slots) in enumerate(self.frames):
                for si, cell in enumerate(slots):
                    sx, sy = self._slot_at(fi, si)
                    if cell is None and abs(x - sx) <= 2 and abs(y - sy) <= 2:
                        slots[si] = self.palette[self.sel]
                        self.palette[self.sel] = None
                        self.sel = None
                        return self._obs()
        return self._obs()


def test_program_solver_fills_the_slots_and_runs():
    env = ProgramRoom()
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
    program = next(s for s in agent.solvers if s.name == "program")
    assert obs.state == GameState.WIN, (done_at, dict(program.stats))
    # three tiles and a run; then 6, a second call and 15 in the first frame, 9 and 12
    # in the subroutine it calls twice: five tiles and a run
    assert done_at == [7, 18], (done_at, dict(program.stats))
