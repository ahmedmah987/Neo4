"""The marker-family solver on three small games, one per way of moving a piece:
buttons that turn a ring of tiles, a button that extends an arm, and keys that
move whichever of two pieces is selected. In each, a piece must cover the
static mark of its own colour; nothing in the solver knows which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.library import LibraryAgent
from neo4.marker import Comp, cover_places, covers, kind_of, match

BG = 4


def blank() -> np.ndarray:
    g = np.full((fp.GRID, fp.GRID), 3, dtype=np.int8)
    g[4:60, 4:60] = BG
    return g


def obs_of(state, done, levels, grid, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = [grid]
    return o


def play(agent, env, steps):
    """Steps at which each level was completed."""
    obs = env.reset()
    out = []
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(out) < obs.levels_completed:
            out.append(used)
        if obs.state == GameState.WIN:
            break
    return out


class RingRoom:
    """Tiles on a rectangular ring; the left button turns it one way, the right
    button the other; the colour-11 tile must stop inside the four colour-11
    corner dots."""

    CELLS = [(10 + 6 * i, 10) for i in range(7)] + [(46, 10 + 6 * i) for i in range(1, 6)] \
        + [(46 - 6 * i, 40) for i in range(1, 7)] + [(10, 40 - 6 * i) for i in range(1, 5)]
    COLORS = [1, 2, 9, 10, 15]
    LAYOUTS = [(3, 0), (12, 16)]  # (index of the goal tile, index of the marked cell)

    def __init__(self, levels: int = 2):
        self.levels = levels
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self._place(0)

    def _place(self, level: int) -> None:
        goal, self.mark = self.LAYOUTS[level % len(self.LAYOUTS)]
        n = len(self.CELLS)
        self.tiles = [self.COLORS[(i * 3 + level) % len(self.COLORS)] for i in range(n)]
        self.tiles[goal] = 11

    def _grid(self) -> np.ndarray:
        g = blank()
        for (x, y), c in zip(self.CELLS, self.tiles):
            g[y:y + 4, x:x + 4] = c
        mx, my = self.CELLS[self.mark]
        for cx, cy in ((mx - 2, my - 2), (mx + 5, my - 2), (mx - 2, my + 5), (mx + 5, my + 5)):
            g[cy, cx] = 11
        g[22:29, 52:55] = 14  # right button
        g[23:28, 55] = 14
        g[22:29, 51] = 4
        g[22:29, 2] = 8  # left button (the mirror image)
        g[23:28, 1] = 8
        g[22:29, 0:2] = 3
        g[22:29, 2:5] = 8
        g[23:28, 1] = 8
        return g

    def _obs(self) -> FrameDataRaw:
        return obs_of(self.state, self.done, self.levels, self._grid(), [6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._place(self.done)
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action is not GameAction.ACTION6 or data is None:
            return self._obs()
        x, y = data["x"], data["y"]
        if 22 <= y <= 28 and 0 <= x <= 5:
            self.tiles = self.tiles[1:] + self.tiles[:1]  # every tile one cell back
        elif 22 <= y <= 28 and 51 <= x <= 56:
            self.tiles = self.tiles[-1:] + self.tiles[:-1]  # every tile one cell on
        else:
            return self._obs()
        if self.tiles[self.mark] == 11:
            self.done += 1
            if self.done == self.levels:
                self.state = GameState.WIN
            else:
                self._place(self.done)
        return self._obs()


class ArmRoom:
    """A colour-11 arm grows down from a cap; its colour-13 tip must reach the
    centre of the four colour-13 dots. The button on the right extends the arm
    by three pixels, the one on the left shortens it."""

    LAYOUTS = [(20, 9), (36, 6)]  # (arm x, extensions needed)

    def __init__(self, levels: int = 2):
        self.levels = levels
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self._place(0)

    def _place(self, level: int) -> None:
        self.x, self.need = self.LAYOUTS[level % len(self.LAYOUTS)]
        self.length = 1

    def _grid(self) -> np.ndarray:
        g = blank()
        x, top = self.x, 10
        g[top, x:x + 3] = 3  # the cap
        bottom = top + 3 * self.length
        g[top + 1:bottom + 1, x:x + 3] = 11
        g[bottom, x + 1] = 13  # the tip
        cy = top + 3 * (1 + self.need)
        g[cy - 1, x + 1] = 13
        g[cy + 1, x + 1] = 13
        g[cy, x] = 13
        g[cy, x + 2] = 13
        g[50:55, 44:49] = 2  # shorten
        g[50:55, 52:57] = 2  # extend
        g[52, 45:48] = 11
        g[51:54, 53:56] = 11
        g[52, 54] = 2
        return g

    def _obs(self) -> FrameDataRaw:
        return obs_of(self.state, self.done, self.levels, self._grid(), [6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._place(self.done)
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED or action is not GameAction.ACTION6 or data is None:
            return self._obs()
        x, y = data["x"], data["y"]
        if 50 <= y <= 54 and 52 <= x <= 56:
            self.length = min(self.length + 1, 14)
        elif 50 <= y <= 54 and 44 <= x <= 48:
            self.length = max(self.length - 1, 1)
        else:
            return self._obs()
        if self.length == 1 + self.need:
            self.done += 1
            if self.done == self.levels:
                self.state = GameState.WIN
            else:
                self._place(self.done)
        return self._obs()


class CrossRoom:
    """Two crosses; arrow keys move the selected one (its centre is black) three
    pixels, ACTION5 selects the other. Each cross must pass over the dots of its
    colour, which are laid out along a row and a column."""

    def __init__(self):
        self.levels = 1
        self.done = 0
        self.state = GameState.NOT_PLAYED
        self._place()

    def _place(self) -> None:
        self.pos = {9: (36, 42), 11: (21, 27)}
        self.sel = 9
        self.goal = {9: (45, 24), 11: (15, 12)}

    def _grid(self) -> np.ndarray:
        g = np.full((fp.GRID, fp.GRID), 5, dtype=np.int8)
        g[63, :] = 15
        for col, (gx, gy) in self.goal.items():
            for mx, my in ((gx, gy - 6), (gx, gy + 6), (gx - 6, gy), (gx + 6, gy)):
                g[my - 1:my + 2, mx - 1:mx + 2] = 4
                g[my, mx] = col
        for col, (cx, cy) in self.pos.items():
            g[cy - 9:cy + 10, cx] = col
            g[cy, cx - 9:cx + 10] = col
            if col == self.sel:
                g[cy, cx] = 0
        return g

    def _obs(self) -> FrameDataRaw:
        return obs_of(self.state, self.done, self.levels, self._grid(), [1, 2, 3, 4, 5])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._place()
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs()
        if action is GameAction.ACTION5:
            self.sel = 11 if self.sel == 9 else 9
            return self._obs()
        d = {GameAction.ACTION1: (0, -3), GameAction.ACTION2: (0, 3),
             GameAction.ACTION3: (-3, 0), GameAction.ACTION4: (3, 0)}.get(action)
        if d is None:
            return self._obs()
        x, y = self.pos[self.sel]
        nx, ny = x + d[0], y + d[1]
        if 10 <= nx <= 53 and 10 <= ny <= 53:
            self.pos[self.sel] = (nx, ny)
        if all(self.pos[c] == self.goal[c] for c in self.pos):
            self.done = 1
            self.state = GameState.WIN
        return self._obs()


def test_matching_reads_a_turning_ring_as_one_step_per_tile():
    room = RingRoom()
    room.reset()
    from neo4.marker import components
    before = components(room._grid(), None)
    for i, c in enumerate(before):
        c.id = i
    room.step(GameAction.ACTION6, {"x": 54, "y": 25})
    after = components(room._grid(), None)
    moved = match(before, after)
    steps = {abs(dx) + abs(dy) for dx, dy in moved.values()}
    assert steps <= {6, 12, 30, 36}, steps  # one cell; wraps only where the ring closes


def test_cover_places_and_kinds():
    rel = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    assert covers(rel, (12, 19), (13.5, 20.5))
    assert (12, 19) in cover_places(rel, (13.5, 20.5))


def test_ring_room_is_solved_by_planning_after_one_probe_per_button():
    steps = play(LibraryAgent(seed=0), RingRoom(), 400)
    assert len(steps) == 2, steps
    # level 2: the goal tile is 4 cells from its mark (16 - 12), the other way 16;
    # the buttons are known kinds by now, so both are probed and the short way taken
    assert steps[1] - steps[0] <= 4 + 7, steps  # four clicks, a few probes and corrections


def test_arm_room_tip_merges_into_the_mark_and_the_level_ends():
    agent = LibraryAgent(seed=0)
    steps = play(agent, ArmRoom(), 400)
    assert len(steps) == 2, (steps, agent.stats)
    assert steps[1] - steps[0] <= 6 + 3, steps
    assert agent.stats.get("marker_levels", 0) >= 1


def test_cross_room_needs_the_selection_key_between_pieces():
    agent = LibraryAgent(seed=0)
    steps = play(agent, CrossRoom(), 600)
    assert len(steps) == 1, (steps, agent.stats)
    assert agent.stats.get("marker_mode_presses", 0) + agent.stats.get("marker_probes", 0) >= 1
