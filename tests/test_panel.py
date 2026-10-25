"""The panel solver on a small game: a board of 3-pixel cells on a background, the panel of
buttons on the left of it. The avatar (a solid square) walks onto open cells only; each button
steps its group of cells to the next of its states (a tile shuts or opens, a bridge turns, a
bar slides along a track or grows over it); a key on a tile shows a hidden button when the
avatar steps on it; a press with the avatar on a pad of the pads' group takes it to the other
pad; a press that leaves the avatar on a cell that is not open is undone (and costs steps).
The level ends on the goal. Colours, sizes, the panel's side and the layouts are none of
dc22's, and the rules here are written apart from the solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, PANEL, LINE, FLOOR, AVATAR, GOAL, KEY, TRACK, BAR = 1, 7, 0, 2, 12, 10, 15, 5, 8
X0, Y0, C = 20, 2, 3  # the board's first cell (pixels) and the cell's size
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
BUDGET = 150


def rect(x1, y1, x2, y2):
    return [(i, j) for j in range(y1, y2 + 1) for i in range(x1, x2 + 1)]


def paint(cells, pattern):
    return {c: pattern for c in cells}


SOLID6, CENTRE = ("solid", 6), ("solid", 13)
# each level: static floor cells, cells of track, the start, the goal, keys (cell -> the group
# it shows), and the groups: a button (colour, shape), the cells of each state, pads
LEVELS = [
    # a tile pair that swaps, and a bridge that turns on its centre; two squares to tell apart
    dict(floor=rect(1, 15, 2, 16) + rect(1, 11, 1, 13) + rect(4, 4, 4, 8) + rect(5, 4, 7, 4) + rect(9, 3, 11, 5),
         start=(1, 16), goal=(10, 4),
         groups=[dict(button=(9, "bar"), states=[{(1, 14): ("check", 9), (8, 4): ("solid", 9)},
                                                 {(1, 14): ("solid", 9), (8, 4): ("check", 9)}]),
                 dict(button=(6, "ell"), states=[{**paint(rect(2, 11, 6, 11), SOLID6), (4, 11): CENTRE},
                                                 {**paint(rect(4, 9, 4, 13), SOLID6), (4, 11): CENTRE}])]),
    # a key shows the gate's button; a bar slides along a track that does not carry the avatar
    dict(floor=rect(1, 1, 2, 2) + rect(3, 2, 6, 2) + rect(2, 3, 2, 9) + rect(7, 9, 9, 9) + rect(11, 8, 12, 10),
         track=rect(3, 9, 6, 9), start=(1, 1), goal=(12, 9), keys={(6, 2): 1},
         groups=[dict(button=(3, "bar"), states=[paint([(3, 9), (4, 9)], ("solid", 3)), paint([(4, 9), (5, 9)], ("solid", 3)),
                                                 paint([(5, 9), (6, 9)], ("solid", 3))]),
                 dict(button=(13, "ell"), hidden=True, states=[{(10, 9): ("check", 13)}, {(10, 9): ("solid", 13)}])]),
    # the avatar starts on a tile that a press shuts; pads take it across
    dict(floor=rect(3, 17, 4, 17) + rect(10, 4, 10, 5) + rect(9, 6, 11, 6),
         start=(2, 17), goal=(10, 6),
         groups=[dict(button=(4, "bar"), pads=[(5, 17), (10, 3)],
                      states=[{(5, 17): ("pad", 14, 11), (10, 3): ("pad", 11, 14)}, {(5, 17): ("pad", 11, 14), (10, 3): ("pad", 14, 11)}]),
                 dict(button=(9, "ell"), states=[{(2, 17): ("solid", 9)}, {(2, 17): ("check", 9)}])]),
    # a bar grows over a gap, a cell at a press; a key on the far side shows the gate's button
    dict(floor=rect(1, 10, 4, 10) + rect(8, 10, 11, 10) + rect(11, 11, 11, 12) + rect(8, 5, 8, 8) + rect(7, 3, 9, 4),
         track=rect(5, 10, 7, 10), start=(1, 10), goal=(8, 4), keys={(11, 12): 1},
         groups=[dict(button=(6, "bar"), states=[{}, paint([(5, 10)], ("solid", 4)), paint(rect(5, 10, 6, 10), ("solid", 4)),
                                                 paint(rect(5, 10, 7, 10), ("solid", 4))]),
                 dict(button=(11, "ell"), hidden=True, states=[{(8, 9): ("check", 11)}, {(8, 9): ("solid", 11)}])]),
]


def obs_of(state, done, levels, frames) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=[1, 2, 3, 4, 6])
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def draw_cell(g, cell, pattern):
    x, y = X0 + cell[0] * C, Y0 + cell[1] * C
    for dy in range(C):
        for dx in range(C):
            kind = pattern[0]
            if kind == "solid":
                v = pattern[1]
            elif kind == "check":
                v = pattern[1] if (dx + dy) % 2 == 0 else BG
            elif kind == "pad":
                v = pattern[1] if (dx + dy) % 2 == 0 else pattern[2]
            else:
                v = TRACK
            g[y + dy, x + dx] = v


def is_open(pattern) -> bool:
    return pattern is not None and pattern[0] in ("solid", "pad")


class PanelRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        lv = LEVELS[self.done]
        self.lv = lv
        self.at = lv["start"]
        self.states = [0] * len(lv["groups"])
        self.keys = dict(lv.get("keys", {}))
        self.shown = {i for i, g in enumerate(lv["groups"]) if not g.get("hidden")}
        self.left = BUDGET

    def _patterns(self) -> dict:
        lv = self.lv
        out = {c: ("solid", FLOOR) for c in lv["floor"]}
        out.update({c: ("track",) for c in lv.get("track", [])})
        out[lv["goal"]] = ("solid", GOAL)
        for g, s in zip(lv["groups"], self.states):
            out.update(g["states"][s])
        return out

    def _grid(self, avatar_colour=AVATAR) -> np.ndarray:
        g = np.full((64, 64), BG, dtype=np.int8)
        g[:62, :16] = PANEL
        g[:, 16:18] = LINE
        for cell, p in self._patterns().items():
            draw_cell(g, cell, p)
        for cell in self.keys:
            x, y = X0 + cell[0] * C, Y0 + cell[1] * C
            g[y + 1, x + 1] = g[y + 1, x + 2] = g[y + 2, x + 1] = KEY
        x, y = X0 + self.at[0] * C, Y0 + self.at[1] * C
        g[y:y + C, x:x + C] = avatar_colour
        for i, grp in enumerate(self.lv["groups"]):
            if i not in self.shown:
                continue
            colour, shape = grp["button"]
            by = 6 + 12 * i
            if shape == "bar":
                g[by:by + 4, 4:12] = colour
            else:
                g[by:by + 6, 4:7] = colour
                g[by + 3:by + 6, 4:12] = colour
        g[62:, :] = 0
        g[62:, :self.left * 64 // BUDGET] = BAR
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, len(LEVELS), frames)

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def _button_at(self, x, y):
        for i, grp in enumerate(self.lv["groups"]):
            if i not in self.shown:
                continue
            by = 6 + 12 * i
            shape = grp["button"][1]
            if shape == "bar" and by <= y < by + 4 and 4 <= x < 12:
                return i
            if shape == "ell" and ((by <= y < by + 6 and 4 <= x < 7) or (by + 3 <= y < by + 6 and 4 <= x < 12)):
                return i
        return None

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        self.left -= 1
        if self.left <= 0:
            self.state = GameState.GAME_OVER
            return self._obs([])
        if action is GameAction.ACTION6:
            i = self._button_at(data["x"], data["y"])
            if i is not None:
                grp = self.lv["groups"][i]
                before, at = self.states[i], self.at
                self.states[i] = (before + 1) % len(grp["states"])
                pads = grp.get("pads")
                if pads and self.at in pads:
                    self.at = pads[1 - pads.index(self.at)]
                if not is_open(self._patterns().get(self.at)):
                    pressed = self._grid()
                    falling = self._grid(LINE)
                    self.states[i], self.at = before, at
                    self.left -= 5
                    return self._obs([pressed, falling, self._grid()])
            return self._obs([self._grid()])
        if action in KEYS:
            d = KEYS[action]
            t = (self.at[0] + d[0], self.at[1] + d[1])
            if is_open(self._patterns().get(t)):
                self.at = t
                if t in self.keys:
                    self.shown.add(self.keys.pop(t))
                if t == self.lv["goal"]:
                    finished = self._grid()
                    self.done += 1
                    if self.done == len(LEVELS):
                        self.state = GameState.WIN
                        return self._obs([finished])
                    self._load()
                    return self._obs([finished, self._grid()])
        return self._obs([self._grid()])


def test_panel_solver_learns_the_buttons_and_walks_to_the_goal():
    env = PanelRoom()
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
    solver = next(s for s in agent.solvers if s.name == "panel")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    # level 1: a move to tell the avatar from the goal (to the right, where three of the
    # squares could go: the bridge's centre is one too, and it is reached on the way and seen
    # not to be the goal), each button pressed round its cycle (2 + 2), then the shortest way
    assert per[0] <= 30, (per, dict(solver.stats))
    # level 2: the key first (the gate's button is hidden), the bar's 3 states and the gate's
    # 2 learned, a step onto the track refused
    assert per[1] <= 40, (per, dict(solver.stats))
    # level 3: the first press drops the avatar, the button is learned from the next tile,
    # and the press on the pad is tried
    assert per[2] <= 16, (per, dict(solver.stats))
    # level 4: the bar's four states, then the key on the far side and the gate
    assert per[3] <= 40, (per, dict(solver.stats))
    assert solver.stats["probed"] == 1 and solver.stats["not_goal"] == 1, dict(solver.stats)
    assert solver.stats["keys_taken"] == 2, dict(solver.stats)
    assert solver.stats["learned_teleport"] == 1, dict(solver.stats)
    assert solver.stats["fell_learning"] == 1, dict(solver.stats)
    assert solver.stats["learned_not_floor"] >= 1, dict(solver.stats)
    assert solver.stats["cycle_3"] == 1 and solver.stats["cycle_4"] == 1, dict(solver.stats)
