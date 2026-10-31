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
    # the copy of the module the agent uses: test_bundle drops the package from sys.modules, so a
    # fresh import here could be another copy
    shared = ModelAgent.__init__.__globals__["shared"]

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
    # the copy of the module the agent uses: test_bundle drops the package from sys.modules, so a
    # fresh import here could be another copy
    shared = ModelAgent.__init__.__globals__["shared"]

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


def test_what_an_action_changed_gives_new_guesses():
    g = np.zeros((64, 64), dtype=np.int8)
    g[20:23, 20:23] = 6  # a block of colour 6
    g[20:23, 40:43] = 6  # and a mark of its colour, hollow
    g[21, 41] = 0
    g[50:52, 10:12] = 3  # two tiles of one shape
    g[50:52, 20:22] = 3
    after = g.copy()
    after[20:23, 20:23] = 0
    after[20:23, 24:27] = 6  # the block moved towards the mark
    after[50:52, 10:12] = 4  # a tile changed colour
    before_objs, after_objs = extract(g), extract(after)
    block = next(o for o in after_objs if o.color == 6 and o.area == 9)
    from neo4.objects import diff
    found = hy.from_events(diff(before_objs, after_objs), after_objs, None, {block.sig})
    keys = {h.key for h in found}
    bring = hy.Hypothesis("bring", (block.sig, ("colour", 6)))
    assert bring.key in keys
    assert ("paint", (block.sig[1:], 3)) not in keys  # the block did not change colour
    tile_shape = next(o for o in after_objs if o.color == 4).sig[1:]
    assert ("paint", (tile_shape, 3)) in keys and ("paint", (tile_shape, 4)) in keys
    assert hy.measure(bring, before_objs, g, None) > hy.measure(bring, after_objs, after, None) > 0
    paint = hy.Hypothesis("paint", (tile_shape, 3))
    assert hy.measure(paint, before_objs, g, None) == 2 and hy.measure(paint, after_objs, after, None) == 1
    # a kind not known to move by what the player does gives no "bring" guess
    assert not any(h.kind == "bring" for h in hy.from_events(diff(before_objs, after_objs), after_objs, None, set()))


def test_the_agent_generates_guesses_from_what_moved_and_tests_them():
    agent = ModelAgent(seed=0)
    agent.generate = True
    g = np.zeros((64, 64), dtype=np.int8)
    g[20:23, 20:23] = 6
    g[20:23, 40:43] = 6
    g[21, 41] = 0
    objs = extract(g)
    agent.hyps = []
    agent.last_objs = objs
    agent.last_grid = g
    agent._generate(objs)  # nothing moved: the block is seen standing still
    after = g.copy()
    after[20:23, 20:23] = 0
    after[20:23, 24:27] = 6
    new = agent._generate(extract(after))
    assert any(h.kind == "bring" and h.params[1] == ("colour", 6) for h in new)
    assert agent.stats["event_hyps"] == len(new) and agent.event_hyps == len(new)


def test_the_record_turns_a_goal_met_too_early_into_a_goal_with_a_condition():
    agent = ModelAgent(seed=0)
    agent.learn_conditions = True
    door = hy.Hypothesis("reach", ((9, 2, 2, "door"),), first=10)
    key = hy.Hypothesis("clear", ((4, 1, 1, "key"),), first=1)
    other = hy.Hypothesis("reach", ((5, 1, 1, "lamp"),), first=6)
    agent.hyps = [door, key, other]
    agent.hyp_values = {door.key: 4, key.key: 1, other.key: 6}
    agent._record({door.key: 0, key.key: 1, other.key: 6}, 1)  # the door touched, the key still there
    agent.hyp_values = {door.key: 0, key.key: 1, other.key: 6}
    agent._record({door.key: 3, key.key: 0, other.key: 6}, 2)  # the key taken
    agent.hyp_values = {door.key: 1, key.key: 0, other.key: 6}  # beside the door when the level ends
    types = agent._against_record([door], 3)
    assert types == [("given", agent._hyp_type(door), agent._hyp_type(key))]

    # the next level: steered to the key first, then to the door
    agent.confirmed = types
    agent.confirm_count[types[0]] += 1
    agent.life_met = set()
    door2 = hy.Hypothesis("reach", ((9, 2, 2, "door"),), first=8)
    key2 = hy.Hypothesis("clear", ((4, 1, 1, "key"),), first=1)
    agent.hyps = [door2, key2]
    agent.hyp_values = {door2.key: 8, key2.key: 1}
    assert agent._steering() == [key2]
    agent.life_met = {key2.key}
    assert agent._steering() == [door2]


def test_the_record_names_the_last_action_or_refutes_the_guess():
    agent = ModelAgent(seed=0)
    agent.learn_conditions = True
    exit_ = hy.Hypothesis("reach", ((9, 2, 2, "exit"),), first=10)
    agent.hyps = [exit_]
    agent.hyp_values = {exit_.key: 2}
    agent._record({exit_.key: 0}, 1)  # reached the exit, the level went on
    agent.hyp_values = {exit_.key: 0}
    agent._record({exit_.key: 0}, 4)  # a move tried while standing on it
    agent.hyp_values = {exit_.key: 0}
    assert agent._against_record([exit_], 5) == [("then", agent._hyp_type(exit_), 5)]  # the key that ended it
    assert agent._against_record([exit_], 4) == []  # tried before without effect: refuted
    assert agent.stats["cond_refuted"] == 1
