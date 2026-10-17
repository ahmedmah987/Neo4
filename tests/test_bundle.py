"""The bundled single-file agent runs inside a stand-in for the ARC-AGI-3-Agents
framework: an `agents` package with the Agent base class, FrameData, GameAction and
GameState shaped like the real ones, driving the local game engine."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from enum import Enum
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def install_framework_shim() -> None:
    """A minimal `agents` package mirroring what MyAgent uses from the real framework."""
    import arcengine

    agents = types.ModuleType("agents")
    structs = types.ModuleType("agents.structs")
    agent_mod = types.ModuleType("agents.agent")

    class GameState(str, Enum):
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class GameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        def __init__(self, value: int) -> None:
            self.action_data: dict = {}
            self.reasoning = None

        def is_complex(self) -> bool:
            return self.value == 6

        def set_data(self, data: dict) -> dict:
            self.action_data = dict(data)
            return self.action_data

        @classmethod
        def from_name(cls, name: str) -> "GameAction":
            return cls[name.upper()]

    class FrameData:
        def __init__(self, **kw) -> None:
            self.game_id = kw.get("game_id", "")
            self.frame = kw.get("frame", [])
            self.state = kw.get("state", GameState.NOT_PLAYED)
            self.levels_completed = kw.get("levels_completed", 0)
            self.win_levels = kw.get("win_levels", 0)
            self.full_reset = kw.get("full_reset", False)
            self.available_actions = kw.get("available_actions", [])

    class Agent:
        MAX_ACTIONS = 80

        def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None) -> None:
            self.card_id, self.game_id, self.agent_name = card_id, game_id, agent_name
            self.arc_env = arc_env
            self.frames = [FrameData(levels_completed=0)]
            self.action_counter = 0

        @property
        def name(self) -> str:
            return self.agent_name

        def _convert(self, raw) -> FrameData:
            return FrameData(
                game_id=raw.game_id,
                frame=[f.tolist() for f in raw.frame],
                state=GameState(raw.state.value),
                levels_completed=raw.levels_completed,
                win_levels=raw.win_levels,
                full_reset=raw.full_reset,
                available_actions=list(raw.available_actions),
            )

        def main(self) -> None:
            while not self.is_done(self.frames, self.frames[-1]) and self.action_counter <= self.MAX_ACTIONS:
                obs = self.arc_env.observation_space
                if obs is not None:
                    self.frames[-1] = self._convert(obs)
                action = self.choose_action(self.frames, self.frames[-1])
                engine_action = arcengine.GameAction.from_name(action.name)
                data = action.action_data if action.is_complex() else None
                raw = self.arc_env.step(engine_action, data=data, reasoning=action.reasoning)
                if raw is not None:
                    self.frames.append(self._convert(raw))
                self.action_counter += 1

    structs.FrameData, structs.GameAction, structs.GameState = FrameData, GameAction, GameState
    agent_mod.Agent = Agent
    agents.structs, agents.agent = structs, agent_mod
    sys.modules["agents"], sys.modules["agents.structs"], sys.modules["agents.agent"] = agents, structs, agent_mod


@pytest.fixture(scope="module")
def bundle_path() -> Path:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "bundle.py")], check=True, cwd=ROOT)
    path = ROOT / "kaggle" / "my_agent.py"
    assert path.exists()
    return path


def test_bundle_plays_a_game_through_the_framework_contract(bundle_path: Path):
    install_framework_shim()
    for name in [m for m in list(sys.modules) if m == "neo4" or m.startswith("neo4.")]:
        del sys.modules[name]  # the bundle must bring its own copy of the package
    spec = importlib.util.spec_from_file_location("my_agent_bundle", bundle_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from neo4.run import arcade

    env = arcade().make("ls20", seed=0)
    assert env is not None
    agent = module.MyAgent(card_id="test", game_id="ls20", agent_name="MyAgent.test",
                           ROOT_URL="http://localhost", record=False, arc_env=env)
    agent.MAX_ACTIONS = 800
    agent.main()
    assert agent.action_counter > 700
    assert agent.frames[-1].levels_completed >= 1, "ls20's first level usually falls within 800 steps"
    assert agent._neo4.stats["nodes"] > 10
    assert "neo4" in agent.name
