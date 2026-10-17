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
STUCK_WALKS = 8  # random walks in a row before the graph is reopened
MAX_REPLAYS = 2  # attempts per level at replaying the previous level's solution


@dataclass(slots=True)
class Candidate:
    action: GameAction
    data: dict[str, int] | None
    group: int
    target: tuple | None = None  # (colour, height, width, shape hash) of the clicked component

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
    passes: list[int] = field(default_factory=list)  # times the edge led somewhere
    deaths: list[int] = field(default_factory=list)  # times the edge ended the game
    interest: int = 0  # untested candidates whose outcome is unknown; set by model agents
    h: float | None = None  # how far the frame is from the learned goal relation; set by model agents

    def __post_init__(self) -> None:
        self.result = [UNTESTED] * len(self.cands)
        self.target = [None] * len(self.cands)
        self.tested_epoch = [-1] * len(self.cands)
        self.passes = [0] * len(self.cands)
        self.deaths = [0] * len(self.cands)
        self.interest = len(self.cands)

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
        # Insertion-ordered (a dict, not a set): BFS order must not depend on the
        # process's string-hash seed, or runs with the same seed would differ.
        self.reverse: dict[str, dict[str, None]] = {}
        self._dist_cache: dict[tuple[int, bool], tuple[dict[str, int], dict[str, str]]] = {}
        self.inconsistent = 0  # same (node, action) led to different targets
        self.epoch = 0
        self._max_group = 0

    def add_node(self, key: str, cands: list[Candidate]) -> Node:
        node = self.nodes.get(key)
        if node is None:
            node = Node(key, cands)
            self.nodes[key] = node
            self._dist_cache.clear()
            if cands:
                self._max_group = max(self._max_group, max(c.group for c in cands))
        return node

    def record(self, src: str, idx: int, dst: str | None) -> None:
        """``dst`` is the node the edge led to, the source itself when nothing
        changed, or None when the game was lost."""
        node = self.nodes[src]
        node.tested_epoch[idx] = self.epoch
        self._dist_cache.clear()
        if dst is None:
            node.deaths[idx] += 1
            if node.result[idx] == TRANSITION and node.passes[idx] > node.deaths[idx]:
                # An edge that led somewhere more often than it killed: the loss
                # came from a move budget or a wandering hazard (tu93 ends every
                # attempt after fifty moves), not from the edge. Keep it for
                # path finding, or the node becomes a dead end that forces a
                # reset on every visit.
                return
            self._downgrade(node, idx)
            return
        if dst == src:
            self._downgrade(node, idx)
            return
        old = node.target[idx]
        if node.result[idx] == TRANSITION and old is not None and old != dst:
            self.inconsistent += 1
            self._unlink(src, old)
        node.passes[idx] += 1
        node.result[idx] = TRANSITION
        node.target[idx] = dst
        self.reverse.setdefault(dst, {})[src] = None

    def _downgrade(self, node: Node, idx: int) -> None:
        old = node.target[idx]
        node.result[idx] = NO_CHANGE
        node.target[idx] = None
        if old is not None:
            self._unlink(node.key, old)

    def _unlink(self, src: str, dst: str) -> None:
        """Drop the reverse link unless another edge of ``src`` still leads to ``dst``."""
        node = self.nodes[src]
        if not any(r == TRANSITION and t == dst for r, t in zip(node.result, node.target)):
            self.reverse.get(dst, {}).pop(src, None)

    def reopen(self) -> None:
        self.epoch += 1
        self._dist_cache.clear()

    def mark_tested(self, key: str, idx: int) -> None:
        """Count a candidate as tested in this epoch without executing it."""
        node = self.nodes[key]
        node.tested_epoch[idx] = self.epoch
        self._downgrade(node, idx)
        self._dist_cache.clear()

    def frontier(self, max_group: int, interesting: bool = False, best_first: bool = False) -> list[str]:
        out = [
            k
            for k, n in self.nodes.items()
            if n.untested(max_group, self.epoch) and (not interesting or n.interest > 0)
        ]
        if best_first:
            # Only the frontier nodes closest to the learned goal relation count:
            # exploration then descends the heuristic instead of fanning out.
            hs = [self.nodes[k].h for k in out if self.nodes[k].h is not None]
            if hs:
                best = min(hs)
                out = [k for k in out if self.nodes[k].h is not None and self.nodes[k].h <= best]
        return out

    def distances(
        self, max_group: int, interesting: bool = False, best_first: bool = False
    ) -> tuple[dict[str, int], dict[str, str]]:
        """Multi-source BFS from the frontier over reversed edges.

        Returns distance-to-frontier and the next hop for every node that
        can reach the frontier. With ``interesting`` only frontier nodes that
        still hold candidates of unknown outcome count; with ``best_first``
        only the frontier nodes with the lowest heuristic.
        """
        cache_key = (max_group, interesting, best_first)
        cached = self._dist_cache.get(cache_key)
        if cached is not None:
            return cached
        dist: dict[str, int] = {}
        nxt: dict[str, str] = {}
        queue: deque[str] = deque()
        for k in self.frontier(max_group, interesting, best_first):
            dist[k] = 0
            queue.append(k)
        while queue:
            v = queue.popleft()
            for u in self.reverse.get(v, ()):
                if u not in dist:
                    dist[u] = dist[v] + 1
                    nxt[u] = v
                    queue.append(u)
        self._dist_cache[cache_key] = (dist, nxt)
        return dist, nxt

    def edge_to(self, src: str, dst: str) -> int | None:
        node = self.nodes[src]
        for i, (r, t) in enumerate(zip(node.result, node.target)):
            if r == TRANSITION and t == dst:
                return i
        return None

    def max_group(self) -> int:
        return self._max_group


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
        self.stuck = 0  # random walks in a row (the frontier is out of reach)
        # Solution replay: the shortest action sequence that completed the previous
        # level, replayed in the next one by action and click-target signature.
        self.solutions: list[list[tuple[int, tuple | None, tuple[int, int] | None]]] = []
        self.replay: list | None = None
        self.replay_pos = 0
        self.replay_tries = 0
        self.replay_last: tuple[str, int] | None = None
        self.stats: dict[str, int] = {
            "nodes": 0,
            "resets": 0,
            "game_overs": 0,
            "back_to_start": 0,
            "random_walks": 0,
            "reopens": 0,
            "inconsistent": 0,
            "replays": 0,
            "replay_steps": 0,
            "replay_aborted": 0,
        }

    # -- observation handling -------------------------------------------------

    def _start_level(self, grid: np.ndarray) -> None:
        self.graph = LevelGraph()
        self.mask = fp.status_bar_mask(fp.segment(grid))
        self.first_key = None
        self.last_key = None
        self.last_idx = None
        self.priority = 0
        self.stuck = 0
        self.replay_tries = 0
        self._begin_replay()

    # -- solution replay ------------------------------------------------------

    def _solution(self) -> list[tuple[int, tuple | None, tuple[int, int] | None]] | None:
        """The shortest known action sequence from the level's first frame to the
        move that completed it: (action id, click-target signature, click point)."""
        if self.first_key is None or self.last_key is None or self.last_idx is None:
            return None
        if self.first_key not in self.graph.nodes:
            return None
        prev: dict[str, tuple[str, int]] = {}
        seen = {self.first_key}
        queue: deque[str] = deque([self.first_key])
        while queue and self.last_key not in seen:
            v = queue.popleft()
            node = self.graph.nodes[v]
            for i, (r, t) in enumerate(zip(node.result, node.target)):
                if r == TRANSITION and t is not None and t not in seen:
                    seen.add(t)
                    prev[t] = (v, i)
                    queue.append(t)
        if self.last_key not in seen:
            return None
        edges: list[tuple[str, int]] = [(self.last_key, self.last_idx)]
        k = self.last_key
        while k != self.first_key:
            k, i = prev[k]
            edges.append((k, i))
        edges.reverse()
        out = []
        for k, i in edges:
            c = self.graph.nodes[k].cands[i]
            point = (c.data["x"], c.data["y"]) if c.data is not None else None
            out.append((c.action.value, c.target, point))
        return out

    def _begin_replay(self) -> None:
        # Only sequences of clicks are replayed: a keyboard route belongs to one
        # layout, and the planner carries what a route taught over to the next.
        if self.solutions and self.replay_tries < MAX_REPLAYS and all(
            point is not None for _, _, point in self.solutions[-1]
        ):
            self.replay = list(self.solutions[-1])
            self.replay_pos = 0
            self.replay_tries += 1
            self.replay_last = None
            self.stats["replays"] += 1
        else:
            self.replay = None

    def _abort_replay(self) -> None:
        if self.replay is not None:
            self.replay = None
            self.stats["replay_aborted"] += 1

    def _replay_step(self, node: Node) -> int | None:
        """The next step of the replay at this node, or None when it is over or the
        level does not match it (no such target, or the last step changed nothing)."""
        if self.replay is None:
            return None
        if self.replay_last is not None:
            k, i = self.replay_last
            if self.graph.nodes[k].result[i] == NO_CHANGE:
                self._abort_replay()
                return None
        if self.replay_pos >= len(self.replay):
            self.replay = None  # fully replayed without a level-up: back to exploration
            return None
        aid, target, point = self.replay[self.replay_pos]
        matches = [i for i, c in enumerate(node.cands) if c.action.value == aid and (c.data is None or c.target == target)]
        if not matches:
            self._abort_replay()
            return None
        if point is not None:
            matches.sort(key=lambda i: abs(node.cands[i].data["x"] - point[0]) + abs(node.cands[i].data["y"] - point[1]))
        idx = matches[0]
        self.replay_pos += 1
        self.replay_last = (node.key, idx)
        self.stats["replay_steps"] += 1
        return idx

    def _key(self, grid: np.ndarray) -> str:
        """State identity of a frame; subclasses may abstract it."""
        return fp.state_key(grid, self.mask)

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
                    cands.append(Candidate(action, {"x": x, "y": y}, g, target_signature(c)))
            else:
                cands.append(Candidate(action, None, 0))
        return cands

    def _reset(self) -> tuple[GameAction, None]:
        self.last_key = self.last_idx = None
        self.since_reset = 0
        self.stats["resets"] += 1
        if self.replay is not None:
            self._abort_replay()
        self._begin_replay()  # a level restart is where a replay can start again
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
            if obs.levels_completed > self.levels_completed >= 0:
                solution = self._solution()
                if solution:
                    self.solutions.append(solution)
            self.levels_completed = obs.levels_completed
            self._start_level(grid)

        key = self._key(grid)
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

    def _pick_local(self, node: Node, local: list[int]) -> int:
        """Which of the node's untested candidates to run; subclasses reorder."""
        return self.rng.choice(local)

    curious = False  # walk towards frontier nodes with unknown-outcome candidates first
    best_first = False  # walk towards the frontier nodes closest to the learned goal relation

    def _choose(self, node: Node) -> int | None:
        idx = self._choose_edge(node)
        if idx is None or self.stats["random_walks"] == self._walks_before:
            self.stuck = 0
        return idx

    _walks_before = 0

    def _choose_edge(self, node: Node) -> int | None:
        self._walks_before = self.stats["random_walks"]
        idx = self._replay_step(node)
        if idx is not None:
            return idx
        top = self.graph.max_group()
        if self.stuck >= STUCK_WALKS:
            # Wandering among known transitions has not brought the frontier within
            # reach (a one-way door, a moving hazard that kept killing on the way):
            # re-test everything rather than wander for the rest of the budget.
            self.graph.reopen()
            self.priority = 0
            self.stuck = 0
            self.stats["reopens"] += 1
        while True:
            local = node.untested(self.priority, self.graph.epoch)
            if local:
                return self._pick_local(node, local)
            if self.curious:
                dist, nxt = self.graph.distances(self.priority, interesting=True)
                if node.key in nxt:
                    idx = self.graph.edge_to(node.key, nxt[node.key])
                    if idx is not None:
                        self.stats["curious_steps"] = self.stats.get("curious_steps", 0) + 1
                        return idx
            if self.best_first:
                dist, nxt = self.graph.distances(self.priority, best_first=True)
                if node.key in nxt:
                    idx = self.graph.edge_to(node.key, nxt[node.key])
                    if idx is not None:
                        self.stats["descents"] = self.stats.get("descents", 0) + 1
                        return idx
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
                return self._pick_local(node, local)
        # Frontier exists but is unreachable from here (non-determinism or a
        # one-way transition): take a random known transition to move around,
        # or ask for a reset if the node is a dead end.
        moves = [i for i, r in enumerate(node.result) if r == TRANSITION]
        if moves:
            self.stats["random_walks"] += 1
            self.stuck += 1
            return self.rng.choice(moves)
        return None


def target_signature(c: fp.Component) -> tuple:
    """(colour, height, width, exact shape hash) of a component: what a click target
    is, independently of where it is."""
    import hashlib

    local = np.zeros((c.y2 - c.y1 + 1, c.x2 - c.x1 + 1), dtype=bool)
    local[c.pixels[:, 0] - c.y1, c.pixels[:, 1] - c.x1] = True
    digest = hashlib.blake2b(np.packbits(local).tobytes(), digest_size=6).hexdigest()
    return (int(c.color), int(local.shape[0]), int(local.shape[1]), digest)


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
