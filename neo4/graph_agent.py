"""Graph-exploration agent for ARC-AGI-3 (reimplementation of arXiv:2512.24156).

Every distinct masked frame is a node. Every candidate action at a node is an
edge that is *untested* until it has been executed once. Candidates carry a
priority group (0 = try first). The agent follows Algorithm 1 of the paper:

1. if the current node has untested candidates with group <= p, run one;
2. otherwise walk the first step of the shortest known path to a node that
   has such candidates (the *frontier*);
3. otherwise raise p and repeat.

Each level gets a fresh graph. Actions that return the level to its first
frame, or end the game, are still recorded as tested, which avoids the
loop the original authors reported in their official run.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp

UNTESTED, TRANSITION, NO_CHANGE = 0, 1, -1


@dataclass(slots=True)
class Candidate:
    action: GameAction
    data: dict[str, int] | None
    group: int

    def describe(self) -> str:
        if self.data is None:
            return self.action.name
        return f"{self.action.name}({self.data['x']},{self.data['y']})"


@dataclass(slots=True)
class Node:
    key: str
    cands: list[Candidate]
    result: list[int] = field(default_factory=list)
    target: list[str | None] = field(default_factory=list)
    tested_epoch: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.result = [UNTESTED] * len(self.cands)
        self.target = [None] * len(self.cands)
        self.tested_epoch = [-1] * len(self.cands)

    def untested(self, max_group: int, epoch: int = 0) -> list[int]:
        return [
            i
            for i, (c, e) in enumerate(zip(self.cands, self.tested_epoch))
            if e < epoch and c.group <= max_group
        ]


class LevelGraph:
    """Nodes, tested edges and shortest paths to the frontier for one level.

    Exploration runs in epochs: a candidate counts as untested until it has
    been executed in the current epoch. ``reopen`` starts a new epoch, which
    re-tests everything in a fresh random order while keeping the known
    transitions for path finding. That is what the agent does when a level's
    graph is exhausted without a level-up, which happens when the state key
    misses hidden state and the recorded transitions are unreliable.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.reverse: dict[str, set[str]] = {}
        self._dist_cache: dict[int, tuple[dict[str, int], dict[str, str]]] = {}
        self.inconsistent = 0  # same (node, action) led to different targets
        self.epoch = 0

    def add_node(self, key: str, cands: list[Candidate]) -> Node:
        node = self.nodes.get(key)
        if node is None:
            node = Node(key, cands)
            self.nodes[key] = node
            self._dist_cache.clear()
        return node

    def record(self, src: str, idx: int, dst: str | None) -> None:
        node = self.nodes[src]
        node.tested_epoch[idx] = self.epoch
        if dst is None or dst == src:
            node.result[idx] = NO_CHANGE
        else:
            old = node.target[idx]
            if node.result[idx] == TRANSITION and old is not None and old != dst:
                self.inconsistent += 1
            node.result[idx] = TRANSITION
            node.target[idx] = dst
            self.reverse.setdefault(dst, set()).add(src)
        self._dist_cache.clear()

    def reopen(self) -> None:
        self.epoch += 1
        self._dist_cache.clear()

    def frontier(self, max_group: int) -> list[str]:
        return [k for k, n in self.nodes.items() if n.untested(max_group, self.epoch)]

    def distances(self, max_group: int) -> tuple[dict[str, int], dict[str, str]]:
        """Multi-source BFS from the frontier over reversed edges.

        Returns distance-to-frontier and the next hop for every node that
        can reach the frontier.
        """
        cached = self._dist_cache.get(max_group)
        if cached is not None:
            return cached
        dist: dict[str, int] = {}
        nxt: dict[str, str] = {}
        queue: deque[str] = deque()
        for k in self.frontier(max_group):
            dist[k] = 0
            queue.append(k)
        while queue:
            v = queue.popleft()
            for u in self.reverse.get(v, ()):
                if u not in dist:
                    dist[u] = dist[v] + 1
                    nxt[u] = v
                    queue.append(u)
        self._dist_cache[max_group] = (dist, nxt)
        return dist, nxt

    def edge_to(self, src: str, dst: str) -> int | None:
        node = self.nodes[src]
        for i, (r, t) in enumerate(zip(node.result, node.target)):
            if r == TRANSITION and t == dst:
                return i
        return None

    def max_group(self) -> int:
        return max((c.group for n in self.nodes.values() for c in n.cands), default=0)


class GraphAgent:
    """Training-free exploration agent. Call ``act`` once per observation."""

    name = "graph"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.graph = LevelGraph()
        self.mask: np.ndarray | None = None
        self.levels_completed = -1
        self.first_key: str | None = None
        self.last_key: str | None = None
        self.last_idx: int | None = None
        self.priority = 0
        self.since_reset = 0  # actions taken since the level (re)started
        self.stats: dict[str, int] = {
            "nodes": 0,
            "resets": 0,
            "game_overs": 0,
            "back_to_start": 0,
            "random_walks": 0,
            "reopens": 0,
            "inconsistent": 0,
        }

    # -- observation handling -------------------------------------------------

    def _start_level(self, grid: np.ndarray) -> None:
        self.graph = LevelGraph()
        self.mask = fp.status_bar_mask(fp.segment(grid))
        self.first_key = None
        self.last_key = None
        self.last_idx = None
        self.priority = 0

    def _candidates(self, grid: np.ndarray, available: list[int]) -> list[Candidate]:
        cands: list[Candidate] = []
        for aid in available:
            if aid == GameAction.RESET.value:
                continue
            action = GameAction.from_id(aid)
            if action.is_complex():
                masked = fp.apply_mask(grid, self.mask)
                comps = fp.segment(masked)
                for c, g in zip(comps, fp.click_groups(comps)):
                    x, y = c.click_point()
                    cands.append(Candidate(action, {"x": x, "y": y}, g))
            else:
                cands.append(Candidate(action, None, 0))
        return cands

    def _reset(self) -> tuple[GameAction, None]:
        self.last_key = self.last_idx = None
        self.since_reset = 0
        self.stats["resets"] += 1
        return GameAction.RESET, None

    def act(self, obs: FrameDataRaw) -> tuple[GameAction, dict[str, int] | None]:
        if obs.state == GameState.GAME_OVER:
            if self.last_key is not None and self.last_idx is not None:
                self.graph.record(self.last_key, self.last_idx, None)
            self.stats["game_overs"] += 1
            return self._reset()
        if obs.state == GameState.NOT_PLAYED or obs.is_empty():
            return self._reset()

        grid = np.asarray(obs.frame[-1])
        if obs.levels_completed != self.levels_completed:
            self.levels_completed = obs.levels_completed
            self._start_level(grid)

        key = fp.state_key(grid, self.mask)
        if self.first_key is None:
            self.first_key = key
        node = self.graph.nodes.get(key)
        if node is None:
            node = self.graph.add_node(key, self._candidates(grid, obs.available_actions))
            self.stats["nodes"] = len(self.graph.nodes)

        if self.last_key is not None and self.last_idx is not None:
            self.graph.record(self.last_key, self.last_idx, key)
            if key == self.first_key and self.last_key != self.first_key:
                self.stats["back_to_start"] += 1
        self.stats["inconsistent"] = self.graph.inconsistent

        idx = self._choose(node)
        if idx is None and self.since_reset == 0 and node.cands:
            # A RESET as the first action of a level is a *full* reset back to
            # level 0 in the engine, so never follow a reset with another one.
            idx = self.rng.randrange(len(node.cands))
        if idx is None:
            return self._reset()
        self.last_key, self.last_idx = key, idx
        self.since_reset += 1
        cand = node.cands[idx]
        return cand.action, cand.data

    # -- Algorithm 1 ----------------------------------------------------------

    def _choose(self, node: Node) -> int | None:
        top = self.graph.max_group()
        while True:
            local = node.untested(self.priority, self.graph.epoch)
            if local:
                return self.rng.choice(local)
            dist, nxt = self.graph.distances(self.priority)
            if node.key in nxt:
                idx = self.graph.edge_to(node.key, nxt[node.key])
                if idx is not None:
                    return idx
            if self.priority < top:
                self.priority += 1
                continue
            break
        if not self.graph.frontier(top):
            # Everything reachable is tested and the level is not solved: the
            # state key must be missing hidden state. Re-test in a new order.
            self.graph.reopen()
            self.priority = 0
            self.stats["reopens"] += 1
            local = node.untested(self.priority, self.graph.epoch)
            if local:
                return self.rng.choice(local)
        # Frontier exists but is unreachable from here (non-determinism or a
        # one-way transition): take a random known transition to move around,
        # or ask for a reset if the node is a dead end.
        moves = [i for i, r in enumerate(node.result) if r == TRANSITION]
        if moves:
            self.stats["random_walks"] += 1
            return self.rng.choice(moves)
        return None


class RandomAgent:
    """Control: uniformly random available action, random click coordinates."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.stats: dict[str, int] = {"resets": 0, "game_overs": 0}

    def act(self, obs: FrameDataRaw) -> tuple[GameAction, dict[str, int] | None]:
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or obs.is_empty():
            self.stats["resets"] += 1
            if obs.state == GameState.GAME_OVER:
                self.stats["game_overs"] += 1
            return GameAction.RESET, None
        choices = [a for a in obs.available_actions if a != GameAction.RESET.value]
        action = GameAction.from_id(self.rng.choice(choices))
        data = None
        if action.is_complex():
            data = {"x": self.rng.randrange(fp.GRID), "y": self.rng.randrange(fp.GRID)}
        return action, data


AGENTS: dict[str, type[Any]] = {GraphAgent.name: GraphAgent, RandomAgent.name: RandomAgent}
