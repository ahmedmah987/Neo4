"""Goal hypotheses (Phase 59), what a click does (Phase 60) and goal experiments (Phase 61)."""

import numpy as np

from neo4 import hypotheses as hy
from neo4.effects import EffectModel, imagine
from neo4.graph_agent import Candidate, Node
from neo4.model_agent import ModelAgent
from neo4.objects import at, extract
from arcengine import GameAction


def board() -> np.ndarray:
    g = np.zeros((64, 64), dtype=np.int8)
    g[10:12, 10:12] = 3  # three tiles of one shape, two colours
    g[10:12, 20:22] = 3
    g[10:12, 30:32] = 4
    g[40:44, 40:44] = 9  # a single object
    return g


def test_hypotheses_read_from_the_screen_and_measured():
    g = board()
    objs = extract(g)
    hyps = hy.propose(objs, g, None)
    kinds = {h.kind for h in hyps}
    assert {"reach", "clear", "unify"} <= kinds
    unify = next(h for h in hyps if h.kind == "unify")
    assert hy.measure(unify, objs, g, None) == 2  # two colours among the tiles
    g2 = g.copy()
    g2[10:12, 30:32] = 3
    assert hy.measure(unify, extract(g2), g2, None) == unify.goal == 1


def test_cause_memory_credits_an_action_and_carries_by_type():
    m = hy.CauseMemory()
    key, typ = ("clear", ((3, 2, 2, "x"),)), ("clear", (2, 2, "x"))
    m.record(key, 1, 3, 2, typ)
    m.record(key, 2, 3, 3, typ)
    assert m.live(key) and m.helped(key)
    assert m.mean_gain(key, 1) == 1.0 and m.mean_gain(key, 2) == 0.0
    other = ("clear", ((5, 2, 2, "x"),))  # the same type in a later level, not tried yet
    assert m.mean_gain(other, 1, typ) == 1.0


def test_a_toggle_is_learned_predicted_and_used_in_imagination():
    g = board()
    model = EffectModel()

    def click(grid, x, y):
        o = at(extract(grid), x, y)
        out = grid.copy()
        cells = out[o.y1:o.y2 + 1, o.x1:o.x2 + 1]
        cells[:] = np.where(cells == 3, 4, 3)
        return o, out

    for x in (10, 20, 30, 10):  # four clicks of one shape, in two colours: one relative template
        o, after = click(g, x, 10)
        model.observe(o, g, after, None)
        g = after
    assert model.known_count() == 1 and model.stats["checked"] >= 1
    o = at(extract(g), 20, 10)
    predicted = model.predict(o, g)
    assert predicted is not None and np.array_equal(predicted, click(g, 20, 10)[1])

    def colours(objs, grid):  # unify the tiles' colours
        return len({ob.color for ob in objs if ob.h == 2 and ob.w == 2}) - 1

    plan = imagine(g, None, model, lambda grid, mask: extract(grid), colours)
    assert plan is not None and len(plan) == 1


def experiment_agent() -> tuple[ModelAgent, hy.Hypothesis, hy.Hypothesis]:
    agent = ModelAgent(seed=0)
    agent.experiment = True
    agent.experiment_after = 0
    agent.experiment_patience = 3
    a = hy.Hypothesis("reach", ("a",), first=10)
    b = hy.Hypothesis("clear", ("b",), first=5)
    agent.hyps = [a, b]
    agent.causes.record(a.key, 1, 10, 9)
    agent.causes.record(b.key, 1, 5, 4)
    return agent, a, b


def test_one_hypothesis_is_tested_at_a_time_and_dropped_when_met_without_a_level_up():
    agent, a, b = experiment_agent()
    agent.hyp_values = {a.key: 5, b.key: 4}
    agent._experiment(agent.hyp_values)
    assert agent.trial is a  # the closer to its goal, as a share of where it began
    assert agent._steering() == [a]
    agent.hyp_values = {a.key: 0, b.key: 4}  # met, and the level did not end
    agent._experiment(agent.hyp_values)
    assert a.key in agent.tried and agent.trial is b and agent.stats["trials_met"] == 1


def test_a_test_that_stops_progressing_is_given_up_and_nodes_are_measured_again():
    agent, a, b = experiment_agent()
    node = Node("k", [Candidate(GameAction.ACTION1, None, 0)])
    agent.graph.nodes["k"] = node
    agent.node_values["k"] = {a.key: 8, b.key: 1}
    agent.hyp_values = {a.key: 5, b.key: 4}
    agent._experiment(agent.hyp_values)
    assert agent.trial is a and node.h == 0.8  # the node measured against the test's hypothesis
    for _ in range(3):
        agent._experiment(agent.hyp_values)  # no new best
    assert agent.trial is b and agent.stats["trials_stuck"] == 1
    assert node.h == 0.2  # and again against the next one
    agent.hyp_values = {a.key: 5, b.key: 0}  # b met without a level-up: a is taken again
    agent._experiment(agent.hyp_values)
    assert agent.trial is a and agent.stalled[a.key] == 1
    for _ in range(5):
        agent._experiment(agent.hyp_values)
    assert agent.trial is a  # twice the patience the second time
    agent.experiment_retries = 2
    agent._experiment(agent.hyp_values)
    assert agent.trial is None and agent.stalled[a.key] == 2  # given up twice: not taken again


def test_without_the_flag_no_experiment_runs():
    agent, a, b = experiment_agent()
    agent.experiment = False
    agent.hyp_values = {a.key: 5, b.key: 4}
    assert agent._steering() == []
