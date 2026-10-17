"""The agent is exercised on a tiny deterministic environment that speaks the
toolkit's FrameDataRaw protocol: a corridor of N cells where ACTION1 moves
right, ACTION2 moves left, ACTION3 does nothing and ACTION4 teleports to the
start. ACTION1 from the last cell completes the level.
"""

import numpy as np
from arcengine import ActionInput, FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.graph_agent import GraphAgent, LevelGraph, Candidate, TRANSITION, NO_CHANGE


class Corridor:
    def __init__(self, n: int = 5, levels: int = 2):
        self.n, self.levels = n, levels
        self.pos, self.done, self.state = 0, 0, GameState.NOT_PLAYED

    def _obs(self) -> FrameDataRaw:
        o = FrameDataRaw(state=self.state, levels_completed=self.done, win_levels=self.levels,
                         available_actions=[0, 1, 2, 3, 4])
        if self.state in (GameState.NOT_FINISHED, GameState.WIN):
            g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
            g[10:14, 10 + 4 * self.pos : 14 + 4 * self.pos] = 9
            g[63, 0 : 10 * (self.done + 1)] = 12  # a status bar that grows per level
            o.frame = [g]
        return o

    def reset(self) -> FrameDataRaw:
        self.pos, self.state = 0, GameState.NOT_FINISHED
        return self._obs()

    def step(self, action: GameAction, data=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state != GameState.NOT_FINISHED:
            return self._obs()
        if action is GameAction.ACTION1:
            if self.pos == self.n - 1:
                self.done += 1
                self.pos = 0
                if self.done == self.levels:
                    self.state = GameState.WIN
            else:
                self.pos += 1
        elif action is GameAction.ACTION2:
            self.pos = max(0, self.pos - 1)
        elif action is GameAction.ACTION4:
            self.pos = 0
        return self._obs()


def run(agent, env, steps):
    obs = env.reset()
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.state == GameState.WIN:
            return used, obs
    return steps, obs


def test_agent_explores_every_edge_and_finishes_corridor():
    env = Corridor(n=5, levels=2)
    agent = GraphAgent(seed=1)
    used, obs = run(agent, env, 400)
    assert obs.state == GameState.WIN
    assert obs.levels_completed == 2
    assert used < 200


def test_graph_marks_back_to_start_actions_as_tested():
    env = Corridor(n=4, levels=1)
    agent = GraphAgent(seed=3)
    # Walk until the first level's graph has been fully explored or the level ends.
    obs = env.reset()
    for _ in range(200):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.state == GameState.WIN:
            break
        g = agent.graph
        if g.nodes and all(not n.untested(4) for n in g.nodes.values()):
            break
    for node in agent.graph.nodes.values():
        for cand, res in zip(node.cands, node.result):
            if cand.action is GameAction.ACTION4:
                assert res in (TRANSITION, NO_CHANGE)  # never left untested


def test_level_graph_bfs_next_hop():
    g = LevelGraph()
    c = lambda: [Candidate(GameAction.ACTION1, None, 0), Candidate(GameAction.ACTION2, None, 0)]
    for k in "abc":
        g.add_node(k, c())
    g.record("a", 0, "b")
    g.record("a", 1, None)
    g.record("b", 0, "c")
    g.record("b", 1, "a")
    # 'c' is the only node with untested actions: frontier = {c}
    dist, nxt = g.distances(0)
    assert dist["c"] == 0 and dist["b"] == 1 and dist["a"] == 2
    assert nxt["a"] == "b" and nxt["b"] == "c"
    assert g.edge_to("a", "b") == 0


class Exhaustible:
    """Two frames, no level-up: ACTION1 toggles between them, ACTION2 does nothing.

    The exploration graph is exhausted after a handful of steps; the agent must
    then keep playing (new epoch) instead of resetting forever.
    """

    def __init__(self):
        self.pos, self.state = 0, GameState.NOT_PLAYED
        self.resets = 0

    def _obs(self):
        o = FrameDataRaw(state=self.state, win_levels=1, available_actions=[0, 1, 2])
        g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
        g[10:14, 10 + 8 * self.pos : 14 + 8 * self.pos] = 9
        o.frame = [g]
        return o

    def reset(self):
        self.resets += 1
        self.pos, self.state = 0, GameState.NOT_FINISHED
        return self._obs()

    def step(self, action, data=None):
        if action is GameAction.RESET:
            return self.reset()
        if action is GameAction.ACTION1:
            self.pos ^= 1
        return self._obs()


def test_exhausted_graph_reopens_instead_of_reset_loop():
    env = Exhaustible()
    agent = GraphAgent(seed=0)
    obs = env.reset()
    consecutive = 0
    for _ in range(200):
        a, d = agent.act(obs)
        consecutive = consecutive + 1 if a is GameAction.RESET else 0
        assert consecutive < 2
        obs = env.step(a, d)
    assert agent.stats["reopens"] >= 1
    assert env.resets < 10


def test_agent_resets_on_game_over_and_marks_edge_dead():
    agent = GraphAgent(seed=0)
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[0:4, 0:4] = 9
    first = FrameDataRaw(state=GameState.NOT_FINISHED, available_actions=[0, 1])
    first.frame = [g]
    a, _ = agent.act(first)
    assert a is GameAction.ACTION1
    over = FrameDataRaw(state=GameState.GAME_OVER, available_actions=[0, 1])
    a, _ = agent.act(over)
    assert a is GameAction.RESET
    node = next(iter(agent.graph.nodes.values()))
    assert node.result[0] == NO_CHANGE
