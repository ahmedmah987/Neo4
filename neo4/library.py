"""The mechanics library: family solvers in front of the model-guided agent.

The public ARC-AGI-3 games fall into a handful of mechanic families (the README
has the classification). For a family the library holds a *solver* that
watches every transition while the ordinary model-guided agent explores,
recognises its family from what it sees, and takes control once it can plan;
it hands control back when its model keeps mispredicting or its plan ends
without completing the level. The model-guided agent plays every game no
solver claims and every level a solver gives back.

A solver has three methods: ``observe(action, data, obs)`` after every
transition (also those it did not choose), ``act(obs)`` returning an action
or None to leave the turn to the fallback, and a ``disabled`` flag it sets
when it is sure the game is not its family.
"""

from __future__ import annotations

from typing import Any

from arcengine import FrameDataRaw, GameAction, GameState

from .marker import MarkerSolver
from .model_agent import ModelAgent

Action = tuple[GameAction, dict[str, int] | None]

FAMILIES: list[type] = [MarkerSolver]


class LibraryAgent:
    """Family solvers first, the model-guided agent for everything else."""

    name = "library"

    def __init__(self, seed: int = 0, families: list[type] | None = None) -> None:
        self.seed = seed
        self.fallback = ModelAgent(seed)
        self.solvers = [cls(seed) for cls in (FAMILIES if families is None else families)]
        self.pending: tuple[GameAction, dict | None] | None = None
        self.controller: Any = None  # the solver that chose the last action, if any
        self.stats: dict[str, Any] = {}

    @property
    def dyn(self):  # the explain command reads the fallback's model
        return self.fallback.dyn

    def _hand_back(self) -> None:
        """The fallback resumes mid-level: it must not record a transition from an
        action it did not choose."""
        fb = self.fallback
        fb.last_cand = None
        fb.last_key = fb.last_idx = None

    def act(self, obs: FrameDataRaw) -> Action:
        action, data = self.pending if self.pending is not None else (None, None)
        for s in self.solvers:
            s.observe(action, data, obs)
        chosen: Action | None = None
        if obs.state == GameState.NOT_FINISHED and not obs.is_empty():
            for s in self.solvers:
                if s.disabled:
                    continue
                chosen = s.act(obs)
                if chosen is not None:
                    self.controller = s
                    break
        if chosen is None:
            if self.controller is not None:
                self._hand_back()
                self.controller = None
            chosen = self.fallback.act(obs)
        self.pending = chosen
        self.stats = dict(self.fallback.stats)
        for s in self.solvers:
            for k, v in s.stats.items():
                self.stats[f"{s.name}_{k}"] = v
        self.stats["solver_steps"] = self.stats.get("solver_steps", 0)
        return chosen
