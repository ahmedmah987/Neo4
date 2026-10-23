"""The copy-family solver on a small painting game: a picture at the top left, a
blank canvas of the same size, a cursor that moves round the canvas, a key that
paints the half of the canvas the cursor faces, and a palette to click. The
level ends when the canvas reproduces the picture; nothing in the solver knows
which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent
from neo4.painter import complete

BG = 4


def obs_of(state, done, levels, grid, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = [grid]
    return o


HALVES = [np.s_[0:5, :], np.s_[:, 5:10], np.s_[5:10, :], np.s_[:, 0:5]]  # facing the top, right, bottom, left


def strokes_to_picture(strokes) -> np.ndarray:
    pic = np.zeros((10, 10), dtype=np.int8)
    for place, colour in strokes:
        pic[HALVES[place]] = colour
    return pic


class CopyRoom:
    """The cursor goes round the canvas (top, right, bottom, left) with keys 3
    and 4; key 5 paints the facing half in the selected colour; clicking a
    palette square selects its colour."""

    CURSOR = [(33, 25), (42, 33), (33, 42), (25, 33)]  # top-left corner of the 3x3 cursor per place
    PALETTE = {9: 40, 12: 46, 0: 52}  # colour -> x of its 4x4 square
    PICTURES = [
        strokes_to_picture([(0, 9)]),
        strokes_to_picture([(1, 9), (2, 0), (3, 12)]),
    ]

    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self.actions = 0

    def _place(self) -> None:
        self.canvas = np.zeros((10, 10), dtype=np.int8)
        self.cursor = 0
        self.colour = 9

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)
        g[4:14, 4:14] = self.PICTURES[self.done % len(self.PICTURES)]
        g[30:40, 30:40] = self.canvas
        x, y = self.CURSOR[self.cursor]
        g[y:y + 3, x:x + 3] = 7
        for c, x in self.PALETTE.items():
            g[2:6, x:x + 4] = c
        g[7, self.PALETTE[self.colour]:self.PALETTE[self.colour] + 4] = 15  # selection mark
        return g

    def _obs(self) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(self.PICTURES), self._grid(), [1, 2, 3, 4, 5, 6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._place()
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs()
        self.actions += 1
        if action is GameAction.ACTION3:
            self.cursor = (self.cursor - 1) % 4
        elif action is GameAction.ACTION4:
            self.cursor = (self.cursor + 1) % 4
        elif action is GameAction.ACTION5:
            self.canvas[HALVES[self.cursor]] = self.colour
        elif action is GameAction.ACTION6 and data is not None:
            for c, x in self.PALETTE.items():
                if x <= data["x"] < x + 4 and 2 <= data["y"] < 6:
                    self.colour = c
        if np.array_equal(self.canvas, self.PICTURES[self.done]):
            self.done += 1
            if self.done == len(self.PICTURES):
                self.state = GameState.WIN
            else:
                self._place()
        return self._obs()


def test_painter_copies_the_pictures():
    env = CopyRoom()
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
    assert len(done_at) == 2, done_at
    painter = next(s for s in agent.solvers if s.name == "painter")
    assert painter.confirmed
    # the second picture needs three strokes in three colours, with every brush
    # known from the first level's survey: a handful of actions
    assert done_at[1] - done_at[0] <= 12, done_at


def test_complete_fills_a_rectangle_and_a_triangle():
    # a half painted over pixels that partly had the colour already: the pixels
    # that changed are sure, the ones that had the colour may belong to it, the
    # ones that had another colour and stayed do not
    inside = np.zeros((10, 10), bool)
    inside[0:5, 3:10] = True
    outside = np.zeros((10, 10), bool)
    outside[5:10, :] = True
    sure, maybe = complete(inside, outside)
    assert (sure == inside).all()
    assert maybe[0:5, 0:3].all() and maybe.sum() == 15
    # painted again in a colour none of the half had: the whole half is sure
    inside[0:5, :] = True
    sure, maybe = complete(inside, outside)
    assert (sure == inside).all() and not maybe.any()
    # a corner triangle seen only in its lower part
    i, j = np.indices((10, 10))
    tri = j <= 9 - i
    inside = tri & (i >= 5)
    outside = ~tri & (i >= 5)
    sure, maybe = complete(inside, outside)
    assert (sure == tri).all() and not maybe.any()
