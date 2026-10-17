"""The bundled single-file agent runs inside a stand-in for the ARC-AGI-3-Agents
framework. The real framework uses the engine's own FrameData / GameAction / GameState
(the starter kit's template imports them from arcengine), so the shim only has to
reproduce the Agent base class and its main() loop: convert the raw observation to a
FrameData with list frames, call choose_action, and step the environment with
action.action_data.model_dump() and action.reasoning, as agents/agent.py does."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest
from arcengine import FrameData, GameAction

ROOT = Path(__file__).resolve().parent.parent


def install_framework_shim() -> None:
    agents = types.ModuleType("agents")
    agent_mod = types.ModuleType("agents.agent")

    class Agent:
        MAX_ACTIONS = 80

        def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None) -> None:
            self.card_id, self.game_id, self.agent_name = card_id, game_id, agent_name
            self.arc_env = arc_env
            self.frames = [FrameData(levels_completed=0)]
            self.action_counter = 0
            self.steps_with_reasoning = 0

        @property
        def name(self) -> str:
            return self.agent_name

        @staticmethod
        def _convert(raw) -> FrameData:
            return FrameData(
                game_id=raw.game_id,
                frame=[f.tolist() for f in raw.frame],
                state=raw.state,
                levels_completed=raw.levels_completed,
                win_levels=raw.win_levels,
                guid=raw.guid,
                full_reset=raw.full_reset,
                available_actions=raw.available_actions,
            )

        def main(self) -> None:
            while not self.is_done(self.frames, self.frames[-1]) and self.action_counter <= self.MAX_ACTIONS:
                obs = self.arc_env.observation_space
                if obs is not None:
                    self.frames[-1] = self._convert(obs)
                action: GameAction = self.choose_action(self.frames, self.frames[-1])
                data = action.action_data.model_dump()
                reasoning = getattr(action, "reasoning", None)
                if reasoning is not None:
                    self.steps_with_reasoning += 1
                raw = self.arc_env.step(action, data=data, reasoning=reasoning)
                if raw is not None:
                    self.frames.append(self._convert(raw))
                self.action_counter += 1

    agent_mod.Agent = Agent
    agents.agent = agent_mod
    sys.modules["agents"], sys.modules["agents.agent"] = agents, agent_mod


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
    assert agent.steps_with_reasoning == agent.action_counter
    assert agent.frames[-1].levels_completed >= 1, "ls20's first level usually falls within 800 steps"
    assert agent._neo4.stats["nodes"] > 10
    assert "neo4" in agent.name


def test_bundle_click_actions_carry_coordinates(bundle_path: Path):
    install_framework_shim()
    spec = importlib.util.spec_from_file_location("my_agent_bundle2", bundle_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from neo4.run import arcade

    env = arcade().make("vc33", seed=0)  # a click game
    agent = module.MyAgent(card_id="test", game_id="vc33", agent_name="MyAgent.test",
                           ROOT_URL="http://localhost", record=False, arc_env=env)
    agent.MAX_ACTIONS = 60
    agent.main()
    data = GameAction.ACTION6.action_data.model_dump()
    assert {"x", "y"} <= set(data) and 0 <= data["x"] <= 63 and 0 <= data["y"] <= 63
    assert agent.frames[-1].levels_completed >= 1
