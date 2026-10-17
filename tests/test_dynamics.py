import numpy as np
from arcengine import GameAction

from neo4 import frame as fp
from neo4.dynamics import GAME_OVER, OK, Dynamics
from neo4.objects import extract


def world(ax: int, ay: int = 20) -> np.ndarray:
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[:, 30:32] = 3  # wall
    g[10:14, 40:44] = 11  # lava block, right of the wall
    g[ay : ay + 2, ax : ax + 2] = 9  # avatar 2x2
    return g


def walk_right(dyn: Dynamics, start: int, steps: int, step_px: int = 2, wall_x: int = 30):
    x = start
    for _ in range(steps):
        before = world(x)
        nx = x + step_px if x + step_px + 1 < wall_x else x
        after = world(nx)
        dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(after), OK)
        x = nx
    return x


def test_learns_avatar_direction_and_blocking_wall():
    dyn = Dynamics()
    walk_right(dyn, 10, 12)  # moves until the wall, then bumps into it
    avatar = dyn.avatar()
    assert avatar is not None and avatar[:3] == (9, 2, 2)
    assert dyn.direction(GameAction.ACTION1.value) == (2, 0)
    grid = world(28)
    assert dyn.predict(GameAction.ACTION1, None, extract(grid), grid).verdict == "blocked"
    grid = world(10)
    assert dyn.predict(GameAction.ACTION1, None, extract(grid), grid).verdict == "move"
    assert dyn.predict(GameAction.ACTION2, None, extract(grid), grid).verdict == "unknown"
    text = "\n".join(dyn.describe())
    assert "ACTION1: moves it by (2, 0)" in text and "colour 3 blocks the avatar" in text


def test_learns_hazards():
    dyn = Dynamics()
    walk_right(dyn, 10, 4)
    for _ in range(3):
        before = world(38, 10)  # lava directly ahead
        dyn.observe(GameAction.ACTION1, None, extract(before), before, None, GAME_OVER)
    grid = world(38, 10)
    p = dyn.predict(GameAction.ACTION1, None, extract(grid), grid)
    assert p.verdict == "danger"


def test_click_noop_rule():
    dyn = Dynamics()
    g = world(10)
    g[50:54, 50:54] = 6  # a button that does nothing
    objs = extract(g)
    for _ in range(3):
        dyn.observe(GameAction.ACTION6, {"x": 51, "y": 51}, objs, g, extract(g), OK)
    assert dyn.predict(GameAction.ACTION6, {"x": 52, "y": 52}, objs, g).verdict == "noop_click"
    assert dyn.predict(GameAction.ACTION6, {"x": 10, "y": 20}, objs, g).verdict == "unknown"
    changed = g.copy()
    changed[50:54, 50:54] = 8
    dyn.observe(GameAction.ACTION6, {"x": 51, "y": 51}, objs, g, extract(changed), OK)
    assert dyn.click[objs_sig(objs, 6)]["change"] == 1


def test_click_death_rule():
    dyn = Dynamics()
    g = world(10)
    g[50:54, 50:54] = 6  # a trap
    objs = extract(g)
    dyn.observe(GameAction.ACTION6, {"x": 51, "y": 51}, objs, g, None, GAME_OVER)
    assert dyn.predict(GameAction.ACTION6, {"x": 52, "y": 52}, objs, g).verdict == "unknown"
    dyn.observe(GameAction.ACTION6, {"x": 51, "y": 51}, objs, g, None, GAME_OVER)
    p = dyn.predict(GameAction.ACTION6, {"x": 52, "y": 52}, objs, g)
    assert p.verdict == "danger" and "lost the game 2/2" in p.why


def test_click_rules_distinguish_instances_of_one_signature():
    dyn = Dynamics()
    g = world(10)
    g[50:54, 50:54] = 6  # a trap ...
    g[50:54, 58:62] = 6  # ... and a harmless twin with the same signature
    objs = extract(g)
    for _ in range(2):
        dyn.observe(GameAction.ACTION6, {"x": 51, "y": 51}, objs, g, None, GAME_OVER)
    for _ in range(6):
        dyn.observe(GameAction.ACTION6, {"x": 59, "y": 51}, objs, g, extract(g), OK)
    assert dyn.predict(GameAction.ACTION6, {"x": 52, "y": 52}, objs, g).verdict == "danger"
    # The twin is not dangerous, and a no-op is only ever concluded per kind, not per cell.
    assert dyn.predict(GameAction.ACTION6, {"x": 60, "y": 52}, objs, g).verdict == "unknown"


def objs_sig(objs, color):
    return next(o.sig for o in objs if o.color == color)


def test_action_level_death_rule():
    dyn = Dynamics()
    g = world(10)
    objs = extract(g)
    for _ in range(3):
        dyn.observe(GameAction.ACTION5, None, objs, g, None, GAME_OVER)
    assert dyn.predict(GameAction.ACTION5, None, objs, g).verdict == "danger"
    assert dyn.predict(GameAction.ACTION1, None, objs, g).verdict == "unknown"
    assert "ACTION5 lost the game 3/3 times" in "\n".join(dyn.describe())
