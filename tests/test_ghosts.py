"""The replay solver on a small game: corridors of 4-pixel cells on a background, the avatar a
solid square. Wires join plates (a small square in a cell) to doors (a cell filled with the
wire's colour) or to two pads (cells framed in it). A door is open while something stands on
one of its plates, or, for the turning colour, turns at each step onto its plate; a plate
wired to pads swaps what stands on them. The fifth key starts the run again from the start
and leaves a ghost that walks the old run, a move for each of the avatar's; a row of icons
counts the runs. On the last level an enemy walks a track to and fro (back at its ends and
at a shut door), presses plates, and catches the avatar by stepping onto it. Colours, sizes,
layouts and the drawing are none of g50t's, and the rules here are written apart from the
solver's."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, FLOOR, AVATAR, GHOST, ENEMY, HOLD, TURN, SWAP, UNUSED = 3, 7, 12, 6, 13, 10, 15, 4, 1
X0, Y0, S = 3, 8, 4  # the first cell's pixel and the cell's size; cells are S + 1 apart
KEYS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
BUDGET = 120  # moves in one go at a level


def row(j, i1, i2):
    return [(i, j) for i in range(i1, i2 + 1)]


def px(cell):
    return X0 + cell[0] * (S + 1), Y0 + cell[1] * (S + 1)


# each level: open cells, start, goal, runs, groups (colour, kind, plates, doors, pads, the
# wire as a pixel polyline), and maybe an enemy (its track, where it starts on it, its way)
LEVELS = [
    # a ghost on the plate holds the door open
    dict(open=row(1, 1, 7) + [(3, 2), (3, 3)], start=(1, 1), goal=(7, 1), runs=2,
         groups=[dict(colour=HOLD, kind="hold", plates=[(3, 3)], doors=[(5, 1)], pads=[],
                      wire=[(21, 25), (30, 25), (30, 17)])]),
    # two ghosts on two plates, and a door that turns at each step onto its plate
    dict(open=row(5, 1, 10) + [(2, 4), (3, 6), (7, 4)], start=(1, 5), goal=(10, 5), runs=3,
         groups=[dict(colour=TURN, kind="turn", plates=[(2, 4)], doors=[(4, 5)], pads=[],
                      wire=[(14, 28), (14, 25), (25, 25), (25, 32)]),
                 dict(colour=HOLD, kind="hold", plates=[(3, 6)], doors=[(6, 5)], pads=[],
                      wire=[(19, 41), (19, 44), (35, 44), (35, 37)]),
                 dict(colour=HOLD, kind="hold", plates=[(7, 4)], doors=[(8, 5)], pads=[],
                      wire=[(39, 28), (39, 25), (45, 25), (45, 32)])]),
    # the ghost's step onto the plate swaps the avatar from one pad to the other
    dict(open=[(1, 8), (2, 8), (3, 8), (3, 9), (1, 7), (1, 6), (1, 5), (2, 5), (3, 5), (7, 9), (7, 8), (8, 8), (9, 8)],
         start=(1, 8), goal=(9, 8), runs=2,
         groups=[dict(colour=SWAP, kind="swap", plates=[(3, 5)], doors=[], pads=[(3, 9), (7, 9)],
                      wire=[(21, 35), (29, 35), (29, 59), (20, 59), (20, 58), (20, 59), (40, 59), (40, 58)])]),
    # an enemy walks to and fro over a turning plate; its track ends short of its corridor's end
    dict(open=row(2, 2, 9) + row(6, 1, 9), start=(1, 6), goal=(9, 6), runs=2,
         groups=[dict(colour=TURN, kind="turn", plates=[(4, 2)], doors=[(6, 6)], pads=[],
                      wire=[(24, 21), (24, 30), (35, 30), (35, 37)])],
         enemy=dict(track=row(2, 2, 8), at=1, way=1)),
]


def obs_of(state, done, frames) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=len(LEVELS), available_actions=[1, 2, 3, 4, 5])
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


class GhostRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_FINISHED

    def _load(self) -> None:
        self.lv = LEVELS[self.done]
        self.runs: list = []
        self.way = self.lv["enemy"]["way"] if "enemy" in self.lv else 0
        self.left = BUDGET
        self._begin()

    def _begin(self) -> None:
        """Everything back at the start; a ghost for each run kept; the enemy keeps its way."""
        lv = self.lv
        self.at = lv["start"]
        self.moves: list = []
        self.ghosts = [lv["start"] for _ in self.runs]
        self.foe = lv["enemy"]["at"] if "enemy" in lv else None
        self.foe_alive = "enemy" in lv
        self.pressed = [False] * len(lv["groups"])
        self.turned = [False] * len(lv["groups"])

    def _foe_cell(self):
        return self.lv["enemy"]["track"][self.foe] if self.foe_alive else None

    def _door_open(self, gi) -> bool:
        g = self.lv["groups"][gi]
        return self.turned[gi] if g["kind"] == "turn" else self.pressed[gi]

    def _passable(self, cell) -> bool:
        if cell not in self.lv["open"]:
            return False
        return all(self._door_open(gi) for gi, g in enumerate(self.lv["groups"]) if cell in g["doors"])

    def _grid(self) -> np.ndarray:
        lv = self.lv
        g = np.full((64, 64), BG, dtype=np.int64)
        for c in lv["open"]:
            x, y = px(c)
            g[y - 1:y + S + 1, x - 1:x + S + 1] = FLOOR
        for gi, grp in enumerate(lv["groups"]):
            pts = grp["wire"]
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                g[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] = grp["colour"]
            for c in grp["plates"]:
                x, y = px(c)
                g[y + 1:y + 3, x + 1:x + 3] = grp["colour"]
            for c in grp["pads"]:
                x, y = px(c)
                g[y - 1:y + S + 1, x - 1:x + S + 1] = grp["colour"]
                g[y:y + S, x:x + S] = FLOOR
            if not self._door_open(gi):
                for c in grp["doors"]:
                    x, y = px(c)
                    g[y:y + S, x:x + S] = grp["colour"]
        x, y = px(lv["goal"])
        g[y - 1, x - 1:x + S + 1] = AVATAR
        g[y + S, x - 1:x + S + 1] = AVATAR
        g[y:y + S, x + S] = AVATAR
        for i in range(lv["runs"]):
            colour = GHOST if i < len(self.runs) else AVATAR if i == len(self.runs) else UNUSED
            g[1:4, 44 + 4 * i:47 + 4 * i] = colour
        for c, colour in [(gh, GHOST) for gh in self.ghosts] + [(self._foe_cell(), ENEMY), (self.at, AVATAR)]:
            if c is not None:
                x, y = px(c)
                g[y:y + S, x:x + S] = colour
        return g

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, frames)

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def _enemy_step(self):
        """On along the track, back at its ends and from a shut door; the cell it ends on, or
        None when it cannot move."""
        track = self.lv["enemy"]["track"]
        for _ in range(2):
            nxt = self.foe + self.way
            if 0 <= nxt < len(track) and self._passable(track[nxt]):
                self.foe = nxt
                return track[nxt]
            self.way = -self.way
        return None

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        lv = self.lv
        if action is GameAction.ACTION5:
            if self.moves:
                self.runs = self.runs + [self.moves] if len(self.runs) + 1 < lv["runs"] else []
                self._begin()
            return self._obs([self._grid()])
        d = KEYS[action]
        to = (self.at[0] + d[0], self.at[1] + d[1])
        if not self._passable(to):
            return self._obs([self._grid()])
        self.left -= 1
        t = len(self.moves)
        self.moves.append(d)
        self.at = to
        for j, run in enumerate(self.runs):
            if t < len(run):
                gt = (self.ghosts[j][0] + run[t][0], self.ghosts[j][1] + run[t][1])
                if self._passable(gt):
                    self.ghosts[j] = gt
        if self.foe_alive and self._enemy_step() == self.at:
            self.state = GameState.GAME_OVER
            return self._obs([])
        for gi, grp in enumerate(lv["groups"]):
            standing = [self.at] + self.ghosts + ([self._foe_cell()] if self.foe_alive else [])
            now = any(c in grp["plates"] for c in standing)
            if now and not self.pressed[gi]:
                if grp["kind"] == "turn":
                    self.turned[gi] = not self.turned[gi]
                if grp["kind"] == "swap":
                    a, b = grp["pads"]
                    other = {a: b, b: a}
                    self.at = other.get(self.at, self.at)
                    self.ghosts = [other.get(c, c) for c in self.ghosts]
                    if self.foe_alive and self._foe_cell() in other:
                        self.foe = lv["enemy"]["track"].index(other[self._foe_cell()])
            self.pressed[gi] = now
        if not self._passable(self.at):
            self.state = GameState.GAME_OVER  # a door shut on the avatar
            return self._obs([])
        if self.foe_alive and not self._passable(self._foe_cell()):
            self.foe_alive = False
        if self.at == lv["goal"]:
            finished = self._grid()
            self.done += 1
            if self.done == len(LEVELS):
                self.state = GameState.WIN
                return self._obs([finished])
            self._load()
            return self._obs([finished, self._grid()])
        if self.left <= 0:
            self.state = GameState.GAME_OVER
            return self._obs([])
        return self._obs([self._grid()])


def test_replay_solver_plans_ghosts_doors_swaps_and_the_enemy():
    env = GhostRoom()
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
    solver = next(s for s in agent.solvers if s.name == "replay")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    # level 1: the ghost's run onto the plate (4 moves), the fifth key, and the last run, which
    # waits a step and back for the door the ghost opens at its fourth move
    assert per[0] <= 13, (per, dict(solver.stats))
    # level 2: under doors that hold, three plates would need holding with two ghosts: a plate
    # of each colour is stepped on and off (one door shuts, the other stays open: it turns),
    # then a restart, the two ghosts' runs (the second one turns the door on its way) and the
    # last run
    assert per[1] <= 35, (per, dict(solver.stats))
    # level 3: the ghost's run onto the plate, and the last run onto the pad as it is pressed
    assert per[2] <= 14, (per, dict(solver.stats))
    # level 4: the enemy's first step shows its way and its sixth that its track ends short of
    # the corridor's end; the door it turns is open as the avatar passes
    assert per[3] <= 8, (per, dict(solver.stats))
    assert solver.stats["rewinds"] == 4, dict(solver.stats)
    assert solver.stats["explore"] == 2, dict(solver.stats)
    assert solver.stats["learned_hold"] == 1 and solver.stats["learned_turn"] == 1, dict(solver.stats)
    assert solver.stats["learned_heading"] == 1 and solver.stats["learned_off_track"] == 1, dict(solver.stats)
    assert solver.stats["mismatch"] == 0, dict(solver.stats)
