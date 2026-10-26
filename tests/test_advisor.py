"""The goal advisor (Phase 58): answers parsed defensively, the frame rendered, advised objects
clicked or planned to first, and no change at all without a model."""

import json

import numpy as np
from arcengine import GameAction

from neo4.advisor import Advisor, ScriptedAdvisor, build_prompt, parse, render
from neo4.model_agent import ModelAgent
from neo4.run import arcade


def test_parse_keeps_valid_object_numbers_only():
    a = parse('Sure! {"goal": "reach the door", "player": "#2", "targets": [3, "5", 99, 3, "x"]} done', 10)
    assert a is not None and a.goal == "reach the door" and a.player == 2 and a.targets == [3, 5]
    assert parse("no json here", 10) is None
    assert parse('{"targets": 4}', 10).targets == [4]
    assert parse('{"goal": "x", "player": null, "targets": []}', 3).player is None


def test_render_and_prompt():
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10:12, 20:22] = 8
    img = render(grid)
    assert img.shape == (512, 512, 3) and tuple(img[10 * 8 + 3, 20 * 8 + 3]) == (0xF9, 0x3C, 0x31)
    p = build_prompt(["#1: red, 2x2 cells at x=20..21, y=10..11"], [1, 2, 6], ["The keys move #1."], 0)
    assert "level 1" in p and "ACTION6 (click a cell)" in p and "#1: red" in p and '"targets"' in p


def play(agent, game: str, steps: int):
    env = arcade().make(game, seed=0)
    obs = env.reset()
    actions = []
    for _ in range(steps):
        action, data = agent.act(obs)
        actions.append((action.value, json.dumps(data, sort_keys=True)))
        obs = env.step(action, data or {})
    return actions


def first_target(prompt: str) -> str:
    return '{"goal": "click it", "player": null, "targets": [1, 2]}'


def test_an_advised_object_is_clicked_first():
    agent = ModelAgent(seed=0)
    agent.advisor = ScriptedAdvisor(first_target)
    actions = play(agent, "vc33", 3)
    assert agent.stats["advice_queries"] >= 1 and agent.stats["advice_answers"] >= 1
    assert agent.stats["advice_clicks"] >= 1
    assert actions[1][0] == GameAction.ACTION6.value  # the first action after the reset is the advised click
    assert "objects on the screen" in agent.advisor.prompts[0].lower()


def test_the_avatar_is_walked_to_an_advised_object():
    agent = ModelAgent(seed=0)
    agent.advisor = ScriptedAdvisor(lambda p: '{"goal": "go there", "player": null, "targets": [1, 2, 3]}')
    play(agent, "ls20", 200)
    assert agent.stats["advice_plans"] >= 1 and agent.stats["advice_steps"] >= 1
    # later questions tell the advisor what the agent has learned
    assert any("The keys move" in p for p in agent.advisor.prompts[1:]) or len(agent.advisor.prompts) == 1


def test_without_a_model_nothing_changes():
    plain = play(ModelAgent(seed=0), "vc33", 60)
    agent = ModelAgent(seed=0)
    agent.advisor = Advisor()  # no model: never available
    assert play(agent, "vc33", 60) == plain
    assert agent.stats["advice_queries"] == 0
