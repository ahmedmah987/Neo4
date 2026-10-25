"""The mirror solver on a small game: a board of 5-pixel cells inside a frame, lines across it,
pieces, the pieces' reflections in the lines (and in the reflections of the lines), and target
cells. The keys move the selected object a cell (a line only across itself, never off the
board) and cost energy; a click selects the piece under it, or else a line that can move, and
costs none; nothing collides. A level ends when every target is under a piece or a reflection.
The middle pixel of a cell shows the targets' colour where a target is covered, the selection's
colour on every cell of the selected object, and the background on the other objects that can
move. Colours, the cell size, the frame and the layouts are none of ar25's, and the rules here
are written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent
from neo4.mirror import MirrorSolver, find_lattice, read_board

BG, BORDER, LINE, PIECE, IMAGE, TARGET, SEL, BAR = 3, 5, 8, 12, 14, 6, 15, 2
N, C, X0, Y0 = 12, 5, 2, 2  # cells a side, the cell's size, the board's first pixel
BUDGET = 40
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}

LEVELS = [
    # a fixed line; the piece, selected, puts its reflection on the targets
    dict(lines=[("V", 6, True)], pieces=[[(2, 1), (2, 2), (3, 2)]], selected=("P", 0),
         targets=[(7, 9), (7, 10), (6, 10)]),
    # a line that moves, selected, across a fixed one; a piece whose cells touch at corners
    dict(lines=[("V", 3, False), ("H", 9, True)], pieces=[[(6, 2), (7, 3), (8, 2)]], selected=("M", 0),
         targets=[(2, 7), (1, 8), (0, 7), (2, 11), (1, 10), (0, 11)]),
    # no reflection on the board at first; two pieces and two lines that move
    dict(lines=[("V", 11, False), ("H", 0, False)], pieces=[[(8, 8), (9, 8), (8, 9)], [(1, 9), (1, 10)]],
         selected=("M", 1),
         targets=[(4, 7), (3, 7), (4, 8), (8, 3), (9, 3), (8, 2), (4, 3), (3, 3), (4, 2), (11, 9), (11, 10), (11, 1),
                  (11, 0)]),
    # a line mostly under targets
    dict(lines=[("V", 5, False), ("H", 2, False)], pieces=[[(9, 5), (9, 6), (9, 7), (9, 8)]], selected=("M", 0),
         targets=[(5, y) for y in (0, 1, 2, 3, 5, 6, 7, 8)]),
]
OPTIMAL = [11, 7, 13, 5]  # each level's fewest actions, from a search over every final place


def reflect(cells, lines):
    """The cells and their reflections in the lines and in the lines' reflections, on the board."""
    maps, todo = {(1, 0, 1, 0)}, [(1, 0, 1, 0)]
    while todo:
        sx, ax, sy, ay = todo.pop()
        for kind, c, _ in lines:
            m = (-sx, 2 * c - ax, sy, ay) if kind == "V" else (sx, ax, -sy, 2 * c - ay)
            if m not in maps and len(maps) < 64:
                maps.add(m)
                todo.append(m)
    return {(sx * x + ax, sy * y + ay) for sx, ax, sy, ay in maps for x, y in cells
            if 0 <= sx * x + ax < N and 0 <= sy * y + ay < N}


def obs_of(state, done, frames) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=len(LEVELS), available_actions=[1, 2, 3, 4, 6])
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


class MirrorRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        lv = LEVELS[self.done]
        self.lines = [list(m) for m in lv["lines"]]
        self.pieces = [list(p) for p in lv["pieces"]]
        self.selected = lv["selected"]
        self.targets = set(lv["targets"])
        self.left = BUDGET

    def _piece_cells(self):
        return {c for p in self.pieces for c in p}

    def _line_cells(self, m):
        kind, c, _ = m
        return [(c, y) for y in range(N)] if kind == "V" else [(x, c) for x in range(N)]

    def _covered(self) -> bool:
        return self.targets <= reflect(self._piece_cells(), self.lines)

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BORDER, dtype=np.int8)
        g[Y0:Y0 + N * C, X0:X0 + N * C] = BG
        fill, top = {}, {}
        for m in self.lines:
            for q in self._line_cells(m):
                fill[q], top[q] = LINE, ("line",)
        for q in self.targets:
            fill[q], top[q] = TARGET, ("target",)
        pieces = self._piece_cells()
        for q in reflect(pieces, self.lines) - pieces:
            fill[q], top[q] = IMAGE, ("image",)
        for i, p in enumerate(self.pieces):
            for q in p:
                fill[q], top[q] = PIECE, ("piece", i)
        sel = set(self.pieces[self.selected[1]]) if self.selected[0] == "P" else set(self._line_cells(self.lines[self.selected[1]]))
        moving_lines = {q for m in self.lines if not m[2] for q in self._line_cells(m)}
        for (x, y), colour in fill.items():
            px, py = X0 + x * C, Y0 + y * C
            g[py:py + C, px:px + C] = colour
            if (x, y) in self.targets and top[(x, y)][0] in ("piece", "image"):
                mid = TARGET
            elif (x, y) in sel:
                mid = SEL
            elif top[(x, y)][0] == "piece" or (top[(x, y)][0] == "line" and (x, y) in moving_lines):
                mid = BG
            else:
                continue
            g[py + C // 2, px + C // 2] = mid
        g[62:, :] = BORDER
        g[62:, :self.left * 64 // BUDGET] = BAR
        return g

    def _obs(self) -> FrameDataRaw:
        return obs_of(self.state, self.done, [self._grid()])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs()

    def _click(self, x, y) -> None:
        cx, cy = (x - X0) // C, (y - Y0) // C
        if not (0 <= cx < N and 0 <= cy < N and x >= X0 and y >= Y0):
            return
        for i in reversed(range(len(self.pieces))):
            if (cx, cy) in self.pieces[i]:
                self.selected = ("P", i)
                return
        lines = [i for i, m in enumerate(self.lines) if not m[2] and (cx, cy) in self._line_cells(m)]
        if lines:
            others = [i for i in lines if ("M", i) != self.selected]
            self.selected = ("M", (others or lines)[0])

    def _move(self, d) -> None:
        kind, i = self.selected
        if kind == "M":
            m = self.lines[i]
            step = d[0] if m[0] == "V" else d[1]
            if step and 0 <= m[1] + step < N:
                m[1] += step
            return
        moved = [(x + d[0], y + d[1]) for x, y in self.pieces[i]]
        if all(0 <= x < N and 0 <= y < N for x, y in moved):
            self.pieces[i] = moved

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs()
        if action is GameAction.ACTION6:
            self._click(data["x"], data["y"])
        elif action in KEYS:
            self.left -= 1
            self._move(KEYS[action])
        if self._covered():
            self.done += 1
            if self.done == len(LEVELS):
                self.state = GameState.WIN
                return self._obs()
            self._load()
        elif self.left <= 0:
            self.state = GameState.GAME_OVER
        return self._obs()


def test_reader_sees_the_board_of_a_game_unlike_ar25():
    env = MirrorRoom()
    g = np.asarray(env.reset().frame[-1])
    assert find_lattice(g, BG) == (C, X0, Y0, N, N)
    b = read_board(g)
    assert b is not None
    assert b.roles == {"bg": BG, "M": LINE, "P": PIECE, "I": IMAGE, "T": TARGET, "S": SEL}
    assert b.mirrors == [["V", 6, True]] and b.selected == ("P", 0)
    assert [sorted(p.cells()) for p in b.pieces] == [sorted(LEVELS[0]["pieces"][0])]
    assert b.targets == set(LEVELS[0]["targets"])
    # the last level's first line is mostly under targets, and only the line's colour is known
    env.done = 3
    b = read_board(np.asarray(env.reset().frame[-1]), dict(b.roles))
    assert sorted(map(tuple, b.mirrors)) == [("H", 2, False), ("V", 5, False)] and b.selected == ("M", 1)


def test_reader_leaves_other_screens_alone():
    g = np.full((64, 64), BG, dtype=np.int8)
    g[10:20, 10:20] = PIECE  # a block, no lines
    assert read_board(g) is None
    g[:, 30:35] = LINE  # a line, but nothing reflected in it
    assert read_board(g) is None


def test_mirror_solver_moves_lines_and_pieces_to_cover_the_targets():
    env = MirrorRoom()
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
    assert obs.state == GameState.WIN, done_at
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    assert per == OPTIMAL
    solver = next(s for s in agent.solvers if isinstance(s, MirrorSolver))
    assert solver.stats["takeovers"] == len(LEVELS) and not solver.stats["mismatch"], dict(solver.stats)
