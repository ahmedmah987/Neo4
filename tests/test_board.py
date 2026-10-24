"""The clue-board solver on a small game: tiles and clues on a lattice; clicking a
tile turns it to the next colour of the palette (a marked tile also turns the
neighbour its mark points at); a clue's ring says, for each neighbour, the clue's
colour or another one. The first level shows two solved examples (drawings: their
tiles do not turn and their clues are not checked) beside the real clue; the
second shows its palette, whose order is not the order of the colours' numbers.
Nothing in the solver knows which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, SAME, OTHER, MARK = 5, 3, 1, 6
S, O = "same", "other"
RING = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def _around(centre, colour_of):
    """Tiles round a clue: offset -> colour."""
    return {(centre[0] + dx, centre[1] + dy): colour_of(dx, dy) for dx, dy in RING}


class BoardRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        # tiles: place -> [colour, marks, real]; clues: place -> (colour, ring, checked)
        self.tiles: dict = {}
        self.clues: dict = {}
        if self.done == 0:
            self.palette, self.shown = [8, 9], False
            ring_a = dict(zip(RING, [S, O, S, O, O, S, O, S]))
            ring_b = dict(zip(RING, [O, S, O, S, S, O, S, O]))
            ring_r = dict(zip(RING, [O, O, S, S, O, O, S, O]))
            for centre, colour, ring in (((1, 1), 9, ring_a), ((5, 1), 8, ring_b)):
                other = 8 if colour == 9 else 9
                for p, c in _around(centre, lambda dx, dy: colour if ring[(dx, dy)] == S else other).items():
                    self.tiles[p] = [c, (), False]
                self.clues[centre] = (colour, ring, False)
            for p in _around((3, 5), lambda dx, dy: 8):
                self.tiles[p] = [8, (), True]
            self.clues[(3, 5)] = (9, ring_r, True)
        else:
            self.palette, self.shown = [8, 10, 9], True
            for p in [(1, 1), (3, 1), (5, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]:
                self.tiles[p] = [8, (), True]
            self.tiles[(1, 2)][1] = ((1, 0),)  # also turns the tile on its right
            ring1 = {d: O for d in RING}
            ring1.update({(-1, 0): S, (-1, 1): S, (0, 1): S})
            ring2 = {d: O for d in RING}
            ring2.update({(-1, 0): S, (0, 1): S, (1, 1): S})
            self.clues[(2, 1)] = (9, ring1, True)
            self.clues[(4, 1)] = (10, ring2, True)

    @staticmethod
    def _px(p):
        return 4 + 8 * p[0], 4 + 8 * p[1]

    def _grid(self) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)
        for p, (colour, marks, _) in self.tiles.items():
            x, y = self._px(p)
            g[y:y + 6, x:x + 6] = colour
            for dx, dy in marks:
                cx, cy = x + 2 * (dx + 1), y + 2 * (dy + 1)
                g[cy:cy + 2, cx:cx + 2] = MARK
        for p, (colour, ring, _) in self.clues.items():
            x, y = self._px(p)
            for (dx, dy), m in ring.items():
                cx, cy = x + 2 * (dx + 1), y + 2 * (dy + 1)
                g[cy:cy + 2, cx:cx + 2] = SAME if m == S else OTHER
            g[y + 2:y + 4, x + 2:x + 4] = colour
        if self.shown:
            for i, colour in enumerate(self.palette):
                g[4 * i:4 * i + 4, 60:64] = colour
        return g

    def _solved(self) -> bool:
        for p, (colour, ring, checked) in self.clues.items():
            if not checked:
                continue
            for (dx, dy), m in ring.items():
                t = self.tiles.get((p[0] + dx, p[1] + dy))
                if t is None or not t[2]:
                    continue
                if (t[0] == colour) != (m == S):
                    return False
        return True

    def _obs(self, frames=None) -> FrameDataRaw:
        return obs_of(self.state, self.done, 2, (frames or []) + [self._grid()], [6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs()

    def _turn(self, p) -> None:
        t = self.tiles.get(p)
        if t is not None and t[2]:
            t[0] = self.palette[(self.palette.index(t[0]) + 1) % len(self.palette)]

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action is not GameAction.ACTION6:
            return self._obs()
        p = ((data["x"] - 4) // 8, (data["y"] - 4) // 8)
        t = self.tiles.get(p)
        inside = (data["x"] - 4) % 8 < 6 and (data["y"] - 4) % 8 < 6
        if t is None or not t[2] or not inside:
            return self._obs()
        self._turn(p)
        for dx, dy in t[1]:
            self._turn((p[0] + dx, p[1] + dy))
        if self._solved():
            finished = self._grid()
            self.done += 1
            if self.done == 2:
                self.state = GameState.WIN
            else:
                self._load()
            return self._obs([finished])
        return self._obs()


def test_clue_solver_learns_the_marks_from_the_examples_and_follows_the_palette():
    env = BoardRoom()
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
    solver = next(s for s in agent.solvers if s.name == "clue")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    # the three tiles the real clue wants in its colour; none of the examples' tiles
    assert done_at[0] == 3, (done_at, dict(solver.stats))
    assert solver.meaning == {SAME: "same", OTHER: "other"}
    # 8 -> 10 -> 9 as shown: two clicks to 9, one to 10; the marked tile carries its neighbour
    assert done_at[1] - done_at[0] == 7, (done_at, dict(solver.stats))
