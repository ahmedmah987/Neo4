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


def test_experiments_end_a_first_level_that_exploration_takes_thousands_of_actions_to_end():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_energy import EnergyRoom, level_steps

    explorer = ModelAgent(seed=0)
    explorer.experiment = False
    tester = ModelAgent(seed=0)
    first = level_steps(explorer, EnergyRoom(), 4000)[0]
    tested = level_steps(tester, EnergyRoom(), 4000)[0]
    assert tested < 200 < first  # 118 against 3,008: the goal (reach a tile seen once) tested first
    assert any(row[5] == "level" for row in tester.trial_log)


def test_a_copy_rule_is_read_from_clicks_of_two_colours():
    ref = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=np.int8)
    g = np.zeros((64, 64), dtype=np.int8)
    g[2:5, 2:5] = ref  # a picture
    g[20:26, 20:26] = 5  # a board twice its size
    ev = hy.CopyEvidence(g.shape)

    def paint(grid, i, j):
        out = grid.copy()
        out[20 + 2 * i:22 + 2 * i, 20 + 2 * j:22 + 2 * j] = ref[i, j]
        return out

    after = paint(g, 0, 0)
    ev.add(g, after, None)
    assert ev.best() is None  # one click, one colour: it agrees with every place of that colour
    g, after = after, paint(after, 0, 1)
    ev.add(g, after, None)
    rule, gain = ev.best()
    assert rule == ("copy", 2, -8, -8) and gain >= 8
    params = rule + ((20, 20, 6, 6),)
    assert hy.copy_picture_ok(params, after)
    assert hy.copy_measure(params, after) == 36 - 8
    done = after.copy()
    done[20:26, 20:26] = np.kron(ref, np.ones((2, 2), dtype=np.int8))
    assert hy.copy_measure(params, done) == 0


def test_goal_priors_order_the_tests_by_what_goals_were_elsewhere():
    from neo4 import shared

    shared.reset()
    priors = shared.goal_priors()
    goal = hy.Hypothesis("reach", ("x",), features=(("reach", "avatar"), ("reach", "rank", 0)))
    other = hy.Hypothesis("clear", ("y",), features=(("clear", "avatar"), ("clear", "rank", 0)))
    assert priors.score(goal.features) is None  # too few levels yet
    for _ in range(3):  # three levels elsewhere, each ended by a reach guess
        priors.record([goal, other], {goal.key})
    assert priors.score(goal.features) > 0 > priors.score(other.features)

    ModelAgent.share_priors = True
    try:
        agent, a, b = experiment_agent()
    finally:
        ModelAgent.share_priors = False
    a.features, b.features = other.features, goal.features  # a looks like what failed elsewhere
    agent.hyp_values = {a.key: 5, b.key: 4}  # a is the closer to its goal
    agent._experiment(agent.hyp_values)
    assert agent.trial is b  # but b is tested first: guesses like it were goals elsewhere
    shared.reset()


def test_a_situation_seen_elsewhere_sends_the_agent_to_its_first_part_and_back():
    from neo4 import shared

    shared.reset()
    key_like = (("reach", "avatar"), ("reach", "size", "smaller"))
    door_like = (("reach", "avatar"), ("reach", "size", "larger"))
    shared.situations().record(key_like, door_like)  # elsewhere: a key met, then the door ended it
    ModelAgent.share_situations = True
    try:
        agent, a, b = experiment_agent()
    finally:
        ModelAgent.share_situations = False
    a.features, b.features = door_like, key_like  # a is like the door, b like the key
    agent.hyp_values = {a.key: 5, b.key: 4}
    agent._experiment(agent.hyp_values)
    agent.trial = a  # testing the door
    agent.hyp_values = {a.key: 0, b.key: 4}  # the door met, and the level goes on
    agent._experiment(agent.hyp_values)
    assert agent.trial is b and agent.then_key == a.key  # the key first
    agent.hyp_values = {a.key: 3, b.key: 0}  # the key met: back to the door
    agent._experiment(agent.hyp_values)
    assert agent.trial is a and agent.stats["situation_returns"] == 1
    shared.reset()
