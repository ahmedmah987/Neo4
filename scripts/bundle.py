"""Build the single-file Kaggle agent, kaggle/my_agent.py.

The ARC Prize 2026 Kaggle notebook embeds exactly one file, agent/my_agent.py
of the official starter kit, and runs it inside the ARC-AGI-3-Agents framework
(`python main.py --agent myagent`, online mode against a local gateway). This
script packs the `neo4` package into that one file: every module's source is
embedded as a string and materialised as a real module at import time, so the
package code stays byte-for-byte what the tests cover, and a `MyAgent` adapter
translates between the framework's types and the agent's.

    python scripts/bundle.py            # writes kaggle/my_agent.py
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "neo4"
OUT = ROOT / "kaggle" / "my_agent.py"

# Dependency order: each module only imports the ones before it.
MODULES = ["frame", "objects", "planner", "dynamics", "graph_agent", "model_agent", "library"]

HEADER = '''"""Neo4 — ARC-AGI-3 agent for the ARC Prize 2026 Kaggle track (generated file).

Built by scripts/bundle.py from https://github.com/ahmedmah987/Neo4 at commit
{commit} on {date}. Do not edit by hand: edit the package and rebuild.

Drop this file into agent/my_agent.py of the official ARC-AGI-3-Kaggle-Starter
kit; `make play-local` and `make submit` work unchanged.
"""

from __future__ import annotations

import os
import time
import random
import sys
import types

import numpy as np

# -- the neo4 package, embedded -------------------------------------------------

_SOURCES = {sources}

_ORDER = {order}


def _materialise() -> None:
    if "neo4" in sys.modules:
        return
    pkg = types.ModuleType("neo4")
    pkg.__path__ = []  # type: ignore[attr-defined]
    pkg.__package__ = "neo4"
    sys.modules["neo4"] = pkg
    for name in _ORDER:
        full = f"neo4.{{name}}"
        mod = types.ModuleType(full)
        mod.__package__ = "neo4"
        mod.__file__ = f"<neo4-bundle:{{name}}>"
        sys.modules[full] = mod
        exec(compile(_SOURCES[name], mod.__file__, "exec"), mod.__dict__)
        setattr(pkg, name, mod)


_materialise()

from neo4.library import LibraryAgent  # noqa: E402

# -- the framework adapter ------------------------------------------------------
# The ARC-AGI-3-Agents framework uses the engine's own types (the starter kit's
# template imports them the same way), so no translation of enums is needed:
# only the frame, which the framework hands over as nested lists.

from arcengine import FrameData, FrameDataRaw, GameAction, GameState  # noqa: E402
from agents.agent import Agent  # noqa: E402


def _to_raw(latest: FrameData) -> FrameDataRaw:
    """The framework's FrameData (lists) -> the FrameDataRaw (arrays) the agent consumes."""
    raw = FrameDataRaw()
    raw.game_id = getattr(latest, "game_id", "") or ""
    state = getattr(latest, "state", None)
    raw.state = state if isinstance(state, GameState) else GameState(getattr(state, "value", state) or "NOT_PLAYED")
    raw.levels_completed = int(getattr(latest, "levels_completed", 0) or 0)
    raw.win_levels = int(getattr(latest, "win_levels", 0) or 0)
    raw.full_reset = bool(getattr(latest, "full_reset", False))
    raw.available_actions = [int(getattr(a, "value", a)) for a in (getattr(latest, "available_actions", None) or [])]
    frames = getattr(latest, "frame", None) or []
    raw.frame = [np.asarray(f, dtype=np.int8) for f in frames]
    return raw


class MyAgent(Agent):
    """Rule-induction exploration agent; see the repository README for the method.

    In the competition rerun the framework plays every game at once, one thread
    per game in one process, and the notebook has a fixed wall-clock allowance;
    an agent that never stops would time the whole submission out. Two caps:
    ``NEO4_MAX_ACTIONS`` actions per game and ``NEO4_TIME_BUDGET`` seconds of
    wall clock per game, counted from the agent's construction, so that every
    game's agent stops together whatever the number of games.
    """

    MAX_ACTIONS = int(os.environ.get("NEO4_MAX_ACTIONS", "40000"))
    TIME_BUDGET = float(os.environ.get("NEO4_TIME_BUDGET", str(3 * 3600)))

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        seed = random.Random(f"{{self.game_id}}").randrange(1 << 30)
        self._neo4 = LibraryAgent(seed=seed)
        self._steps = 0
        self._started = time.monotonic()

    @property
    def name(self) -> str:
        return f"{{super().name}}.neo4"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Never stop on GAME_OVER: the agent resets the level and carries on.
        if getattr(latest_frame.state, "name", str(latest_frame.state)) == "WIN":
            return True
        return time.monotonic() - self._started > self.TIME_BUDGET

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        obs = _to_raw(latest_frame)
        action, data = self._neo4.act(obs)
        # The framework reads action.action_data.model_dump() and action.reasoning.
        action.set_data(dict(data) if data is not None else {{}})
        self._steps += 1
        action.reasoning = {{
            "step": self._steps,
            "nodes": self._neo4.stats.get("nodes", 0),
            "skipped": self._neo4.stats.get("skipped", 0),
            "plans": self._neo4.stats.get("plans", 0),
            "solver": self._neo4.stats.get("solver", ""),
        }}
        return action
'''


def main() -> None:
    sources = {name: (PKG / f"{name}.py").read_text(encoding="utf-8") for name in MODULES}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    text = HEADER.format(
        commit=commit,
        date=dt.date.today().isoformat(),
        sources="{\n" + "".join(f"    {name!r}: {src!r},\n" for name, src in sources.items()) + "}",
        order=repr(MODULES),
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text):,} bytes, {len(MODULES)} modules, commit {commit})")


if __name__ == "__main__":
    main()
