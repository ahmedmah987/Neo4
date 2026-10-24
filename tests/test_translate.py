"""The translation-family solver on a small game: rules of glyph tiles joined by a
line, an input row and an output row with a cursor; keys 3 and 4 move the cursor,
keys 1 and 2 turn the glyph under it through its family; the level ends when the
output is the input rewritten by the rules. Glyphs are drawn turned at random.
Nothing in the solver knows which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

RULE_BG, ROW_BG, LINE, INK, CURSOR = 2, 3, 3, 5, 0
FRAMES = {"A": 10, "B": 7}
N = 7


def _patterns(seed: int) -> list[np.ndarray]:
    """Seven 5x5 patterns, distinct however they are turned."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < N:
        m = rng.random((5, 5)) < 0.45
        key = min(np.rot90(m, k).tobytes() for k in range(4))
        if key not in seen and m.any() and not m.all():
            seen.add(key)
            out.append(m)
    return out


PATTERNS = {"A": _patterns(1), "B": _patterns(2)}


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


class TranslateRoom:
    # rules (left glyphs, right glyphs), input glyphs, the output's starting glyphs
    LEVELS = [
        ([(["A1"], ["B4"]), (["A2"], ["B6"]), (["A3"], ["B2"])], ["A2", "A3", "A1", "A2"], ["B1", "B1", "B1", "B1"]),
        ([(["A4"], ["B3", "B5"]), (["A5", "A6"], ["B7"]), (["A1"], ["B1"])], ["A5", "A6", "A4", "A1"], ["B2", "B2", "B2", "B2"]),
    ]

    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self.rng = np.random.default_rng(7)

    def _load(self) -> None:
        self.rules, self.source, start = self.LEVELS[self.done]
        self.output = list(start)
        self.cursor = 0
        self.turn = {}  # (place) -> quarter turns of the tile drawn there

    def _tile(self, g, name: str, x: int, y: int) -> None:
        fam, k = name[0], int(name[1:]) - 1
        g[y:y + 7, x:x + 7] = FRAMES[fam]
        turns = self.turn.setdefault((x, y), int(self.rng.integers(4)))
        mask = np.rot90(PATTERNS[fam][k], turns)
        inner = g[y + 1:y + 6, x + 1:x + 6]
        inner[mask] = INK

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), RULE_BG, dtype=np.int8)
        g[34:, :] = ROW_BG
        for r, (left, right) in enumerate(self.rules):
            y = 4 + 10 * r
            x = 4
            for name in left:
                self._tile(g, name, x, y)
                x += 7
            g[y + 3, x:x + 3] = LINE
            x += 3
            for name in right:
                self._tile(g, name, x, y)
                x += 7
        for i, name in enumerate(self.source):
            self._tile(g, name, 8 + 7 * i, 40)
        for i, name in enumerate(self.output):
            self._tile(g, name, 8 + 7 * i, 51)
        cx = 8 + 7 * self.cursor
        g[48:50, cx + 1:cx + 6] = CURSOR
        g[59:61, cx + 1:cx + 6] = CURSOR
        return g

    def _target(self) -> list[str]:
        out, i = [], 0
        while i < len(self.source):
            for left, right in self.rules:
                if self.source[i:i + len(left)] == left:
                    out += right
                    i += len(left)
                    break
            else:
                raise AssertionError("the level has no parse")
        return out

    def _obs(self, frames=None) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(self.LEVELS), (frames or []) + [self._grid()], [1, 2, 3, 4])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs()
        n = len(self.output)
        if action in (GameAction.ACTION3, GameAction.ACTION4):
            self.cursor = (self.cursor + (1 if action is GameAction.ACTION4 else -1)) % n
        elif action in (GameAction.ACTION1, GameAction.ACTION2):
            name = self.output[self.cursor]
            k = (int(name[1:]) - 1 + (1 if action is GameAction.ACTION2 else -1)) % N
            self.output[self.cursor] = name[0] + str(k + 1)
            if self.output == self._target():
                finished = self._grid()
                self.done += 1
                if self.done == len(self.LEVELS):
                    self.state = GameState.WIN
                else:
                    self._load()
                return self._obs([finished])
        return self._obs()


def test_translate_solver_rewrites_the_row_by_the_rules():
    env = TranslateRoom()
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
    solver = next(s for s in agent.solvers if s.name == "translate")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    # at most six turns a glyph (forward, learning the cycle) and the moves between them
    assert done_at[0] <= 4 * 6 + 3, (done_at, dict(solver.stats))
    # the second level with the cycle known: the shorter way round each time
    assert done_at[1] - done_at[0] <= 4 * 3 + 3, (done_at, dict(solver.stats))
