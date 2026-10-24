"""The connector solver on a small game: pieces on a grid of 4-pixel cells, each with
connector cells sticking out of its body; a click selects a piece (a click on the
selected one lets it go), keys 1-4 move it a cell, key 5 turns it a quarter clockwise,
or, for a stack, shows the next of its variants (back and forth). The level is done
when every connector lies on exactly one connector of another piece. On the second
level unselected pieces are greyed out, connectors and all, and one piece is a stack of
three variants of which only the last has a connector. Colours, cell size and shapes
are none of cn04's. Nothing in the solver knows which game it plays."""

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4.library import LibraryAgent

BG, CONN, JOINED, SELECTED, GREY = 1, 7, 6, 5, 9
CELL, N = 4, 16


def obs_of(state, done, levels, frames, actions) -> FrameDataRaw:
    o = FrameDataRaw(state=state, levels_completed=done, win_levels=levels, available_actions=actions)
    if state in (GameState.NOT_FINISHED, GameState.WIN):
        o.frame = frames
    return o


def turn_cw(cells):
    """(body, conn) turned a quarter clockwise within their box."""
    body, conn = cells
    allc = body | conn
    h = max(y for _, y in allc) + 1
    f = lambda p: (h - 1 - p[1], p[0])  # noqa: E731
    return {f(p) for p in body}, {f(p) for p in conn}


SOLVED_B = ({(1, 0), (1, 1), (2, 0)}, {(0, 0), (1, 2)})
START_B = turn_cw(turn_cw(turn_cw(SOLVED_B)))  # one clockwise turn from solved
STACK = [({(0, 0), (1, 0)}, set()), ({(0, 0), (1, 0), (0, 1)}, set()), ({(1, 0), (1, 1)}, {(0, 0)})]


class ConnectRoom:
    def __init__(self) -> None:
        self.done = 0
        self.state = GameState.NOT_PLAYED

    def _load(self) -> None:
        # pieces: dict(x, y, colour, poses (list of (body, conn)), k (pose index), stack)
        if self.done == 0:
            self.grey = False
            self.pieces = [
                dict(x=2, y=2, colour=3, poses=[({(0, 0), (0, 1), (0, 2), (1, 2)}, {(1, 0), (2, 2)})], k=0, stack=False),
                dict(x=9, y=8, colour=4, poses=[START_B], k=0, stack=False),
            ]
        else:
            self.grey = True
            self.pieces = [
                dict(x=2, y=6, colour=3, poses=[({(0, 0), (1, 0), (2, 0)}, {(3, 0)})], k=0, stack=False),
                dict(x=10, y=10, colour=8, poses=list(STACK), k=0, stack=True),
            ]
        self.selected = 0
        self.forward = True

    def _cells(self, p):
        body, conn = p["poses"][p["k"]]
        return ({(p["x"] + x, p["y"] + y) for x, y in body}, {(p["x"] + x, p["y"] + y) for x, y in conn})

    def _grid(self) -> np.ndarray:
        g = np.full((N * CELL, N * CELL), BG, dtype=np.int8)
        conns: dict = {}
        for p in self.pieces:
            for c in self._cells(p)[1]:
                conns[c] = conns.get(c, 0) + 1
        order = [i for i in range(len(self.pieces)) if i != self.selected] + ([self.selected] if self.selected is not None else [])
        for i in order:
            p = self.pieces[i]
            body, conn = self._cells(p)
            sel = i == self.selected
            for (x, y) in body | conn:
                if self.grey and not sel:
                    c = JOINED if (x, y) in conn and conns[(x, y)] == 2 else GREY
                elif (x, y) in conn:
                    c = JOINED if conns[(x, y)] == 2 else CONN
                else:
                    c = SELECTED if sel and not self.grey else p["colour"]
                g[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = c
        return g

    def _solved(self) -> bool:
        conns: dict = {}
        for i, p in enumerate(self.pieces):
            for c in self._cells(p)[1]:
                conns.setdefault(c, []).append(i)
        return bool(conns) and all(len(v) == 2 and v[0] != v[1] for v in conns.values())

    def _obs(self, frames) -> FrameDataRaw:
        return obs_of(self.state, self.done, 2, frames, [1, 2, 3, 4, 5, 6])

    def reset(self) -> FrameDataRaw:
        self.state = GameState.NOT_FINISHED
        self._load()
        return self._obs([self._grid()])

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs([self._grid()])
        if action is GameAction.ACTION6:
            cell = (data["x"] // CELL, data["y"] // CELL)
            order = ([self.selected] if self.selected is not None else []) + [i for i in range(len(self.pieces)) if i != self.selected]
            for i in order:
                body, conn = self._cells(self.pieces[i])
                if cell in body | conn:
                    self.selected = None if i == self.selected else i
                    break
            return self._obs([self._grid()])
        if self.selected is None:
            return self._obs([self._grid()])
        p = self.pieces[self.selected]
        moves = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
        if action in moves:
            dx, dy = moves[action]
            body, conn = p["poses"][p["k"]]
            w = max(x for x, _ in body | conn) + 1
            h = max(y for _, y in body | conn) + 1
            if 0 <= p["x"] + dx and p["x"] + dx + w <= N and 0 <= p["y"] + dy and p["y"] + dy + h <= N:
                p["x"] += dx
                p["y"] += dy
        elif action is GameAction.ACTION5:
            if p["stack"]:
                k = p["k"] + (1 if self.forward else -1)
                if not 0 <= k < len(p["poses"]):
                    self.forward = not self.forward
                    k = p["k"] + (1 if self.forward else -1)
                p["k"] = k
            else:
                p["poses"][p["k"]] = turn_cw(p["poses"][p["k"]])
        else:
            return self._obs([self._grid()])
        if self._solved():
            finished = self._grid()
            self.done += 1
            if self.done == 2:
                self.state = GameState.WIN
                return self._obs([finished])
            self._load()
            return self._obs([finished, self._grid()])
        return self._obs([self._grid()])


def test_connect_solver_joins_the_pieces_turning_and_cycling_variants():
    env = ConnectRoom()
    agent = LibraryAgent(seed=0)
    obs = env.reset()
    done_at = []
    for used in range(1, 150):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        while len(done_at) < obs.levels_completed:
            done_at.append(used)
        if obs.state == GameState.WIN:
            break
    solver = next(s for s in agent.solvers if s.name == "connect")
    assert obs.state == GameState.WIN, (done_at, dict(solver.stats))
    per = [b - a for a, b in zip([0] + done_at, done_at)]
    # select B, one turn, the moves to A (or the same moving A)
    assert per[0] <= 16, (per, dict(solver.stats))
    # see D (a click), learn it is a stack and cycle through it, bring its connector to C's
    assert per[1] <= 24, (per, dict(solver.stats))
    assert solver.stats["stacks"] == 1, dict(solver.stats)
