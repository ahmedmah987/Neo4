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

from .bits import BitSolver
from .board import ClueSolver
from .carry import CarrySolver
from .gravity import GravitySolver
from .connect import ConnectSolver
from .keys import KeySolver
from .marker import MarkerSolver, MarkerSolverNoNav
from .maze import MazeSolver
from .meet import MeetSolver
from .mirror import MirrorSolver
from .spells import SpellSolver
from .painter import PainterSolver
from .panel import PanelSolver
from .pointer import PointerSolver
from .pour import PourSolver
from .program import ProgramSolver
from .replay import ReplaySolver
from .translate import TranslateSolver
from .model_agent import ModelAgent

Action = tuple[GameAction, dict[str, int] | None]

FAMILIES: list[type] = [ProgramSolver, TranslateSolver, ClueSolver, BitSolver, ConnectSolver, CarrySolver, MazeSolver, KeySolver, SpellSolver, PointerSolver, MirrorSolver, MarkerSolver, PainterSolver, MeetSolver, PourSolver, PanelSolver, ReplaySolver, GravitySolver]


class LibraryAgent:
    """Family solvers first, the model-guided agent for everything else."""

    name = "library"

    def __init__(self, seed: int = 0, families: list[type] | None = None) -> None:
        self.seed = seed
        self.fallback = ModelAgent(seed)
        self.solvers = [cls(seed) for cls in (FAMILIES if families is None else families)]
        self.pending: tuple[GameAction, dict | None] | None = None
        self.controller: Any = None  # the solver that chose the last action, if any
        self.resume: Any = None  # the solver whose go ended: first to be asked after the restart
        self.stats: dict[str, Any] = {}
        self.errors: list[str] = []  # solvers switched off by an exception

    @property
    def dyn(self):  # the explain command reads the fallback's model
        return self.fallback.dyn

    def _guarded(self, solver: Any, method: Any, *args: Any) -> Any:
        """A solver that fails on a frame it was not written for is switched off for the
        rest of the game; the other solvers and the explorer play on."""
        try:
            return method(*args)
        except Exception as e:
            solver.disabled = True
            self.errors.append(f"{solver.name}: {e!r}")
            return None

    def _hand_back(self) -> None:
        """The fallback resumes mid-level: it must not record a transition from an
        action it did not choose."""
        fb = self.fallback
        fb.last_cand = None
        fb.last_key = fb.last_idx = None

    def act(self, obs: FrameDataRaw) -> Action:
        action, data = self.pending if self.pending is not None else (None, None)
        for s in self.solvers:
            self._guarded(s, s.observe, action, data, obs)
        chosen: Action | None = None
        if obs.state == GameState.NOT_FINISHED and not obs.is_empty():
            # the solver in control keeps the turn while it wants it, and through the
            # restart that follows a go of its that ended
            first = self.controller or self.resume
            self.resume = None
            order = sorted(self.solvers, key=lambda s: s is not first)
            for s in order:
                if s.disabled:
                    continue
                chosen = self._guarded(s, s.act, obs)
                if chosen is not None:
                    self.controller = s
                    break
        if chosen is None:
            if obs.state == GameState.GAME_OVER and self.controller is not None:
                self.resume = self.controller
            self.controller = None
            chosen = self.fallback.act(obs)
        else:
            # The fallback still sees the step: it records the transition as if it
            # had chosen the action, so its graph and its model stay whole.
            self.fallback.forced = chosen
            self.fallback.act(obs)
        self.pending = chosen
        self.stats = dict(self.fallback.stats)
        for s in self.solvers:
            for k, v in s.stats.items():
                self.stats[f"{s.name}_{k}"] = v
        self.stats["solver_steps"] = self.stats.get("solver_steps", 0)
        if self.errors:
            self.stats["solver_errors"] = list(self.errors)
        return chosen


# The public games each solver was built or tuned on. Documentation of the public set for the
# holdout evaluation, not an input to the agent: a game played without the solvers made for it
# stands in for one of the hidden games, which are other games.
BUILT_FOR: dict[str, list[str]] = {
    "program": ["sb26"], "translate": ["tr87"], "clue": ["ft09"], "bits": ["tn36"],
    "connect": ["cn04"], "carry": ["wa30"], "maze": ["tu93"], "keys": ["ls20"], "spells": ["sc25"],
    "pointer": ["r11l"], "mirror": ["ar25"], "painter": ["cd82"], "meet": ["m0r0"], "pour": ["sp80"],
    "panel": ["dc22"], "replay": ["g50t"], "gravity": ["bp35"],
    "marker": ["lp85", "re86", "s5i5", "r11l", "vc33", "ar25", "su15"],
}


class LibraryHoldoutAgent(LibraryAgent):
    """The library meeting each public game as an unseen one: the solvers built or tuned on the
    game are left out (evaluation only)."""

    name = "library-holdout"
    per_game = True

    def __init__(self, seed: int = 0, game: str = "") -> None:
        super().__init__(seed)
        self.solvers = [s for s in self.solvers if game not in BUILT_FOR.get(s.name, [])]


class LibraryNoNavAgent(LibraryAgent):
    """The library without the Phase 26 navigation additions (ablation)."""

    name = "library-nonav"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed, families=[MarkerSolverNoNav])
