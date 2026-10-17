import numpy as np
from arcengine import GameAction

from neo4 import frame as fp
from neo4.dynamics import GAME_OVER, OK, Dynamics
from neo4.objects import extract, find


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
    return next(o.esig for o in objs if o.color == color)


def test_action_level_death_rule():
    dyn = Dynamics()
    g = world(10)
    objs = extract(g)
    for _ in range(3):
        dyn.observe(GameAction.ACTION5, None, objs, g, None, GAME_OVER)
    assert dyn.predict(GameAction.ACTION5, None, objs, g).verdict == "danger"
    assert dyn.predict(GameAction.ACTION1, None, objs, g).verdict == "unknown"
    assert "ACTION5 lost the game 3/3 times" in "\n".join(dyn.describe())


def test_avatar_prefers_the_unique_sprite_over_a_floor_of_identical_tiles():
    """The avatar covers floor tiles as it moves; the matcher then reports a tile
    'moving' by the opposite vector at every step. The avatar must still win."""
    dyn = Dynamics()

    def scene(ax):
        g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
        for x in range(0, 64, 6):
            g[30:33, x : x + 3] = 7  # a row of identical floor tiles
        g[30:33, ax : ax + 3] = 9  # the avatar sits on the tiles
        return g

    x = 0
    for _ in range(8):
        before, after = scene(x), scene(x + 6)
        dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(after), OK)
        x += 6
    assert dyn.avatar() is not None and dyn.avatar()[0] == 9


def jump_world(ax: int, ay: int = 20) -> np.ndarray:
    """A 3x3 avatar that jumps six cells at a time; a wall two cells thick sits
    between two landing spots, with floor on both sides of it."""
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[:, 44:46] = 3  # the wall: never a landing cell, always in the way
    g[ay : ay + 3, ax : ax + 3] = 9
    return g


def test_jumping_avatar_blames_the_wall_between_not_the_landing_cell():
    dyn = Dynamics()
    x = 2
    for _ in range(6):  # 2 -> 8 -> ... -> 38: clear jumps over the floor
        before, after = jump_world(x), jump_world(x + 6)
        dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(after), OK)
        x += 6
    assert dyn.direction(GameAction.ACTION1.value) == (6, 0)
    for _ in range(3):  # from 38 the jump would cross the wall: the avatar stays put
        before = jump_world(38)
        dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(before), OK)
    assert dyn.cell[3]["blocked"] == 3
    assert dyn.cell[0]["blocked"] == 0 and dyn.cell[0]["passed"] >= 3
    assert dyn.passable(0) and not dyn.passable(3)
    grid = jump_world(38)
    assert dyn.predict(GameAction.ACTION1, None, extract(grid), grid).verdict == "blocked"
    grid = jump_world(8)
    assert dyn.predict(GameAction.ACTION1, None, extract(grid), grid).verdict == "move"


def test_budget_rule_separates_step_limit_losses_from_hazards():
    """Two losses at the longest attempt length ever seen mean the game ends
    every attempt after that many actions: such losses blame nothing."""
    dyn = Dynamics()
    g = world(10)
    objs = extract(g)
    for fifth in (GameAction.ACTION5, GameAction.ACTION3):  # lost at two different places
        for _ in range(4):
            dyn.observe(GameAction.ACTION1, None, objs, g, extract(g), OK)
        dyn.observe(fifth, None, objs, g, None, GAME_OVER)  # fifth action: lost
    assert dyn.budget() == 5 and dyn.last_death_budget
    assert dyn.key_death[GameAction.ACTION5.value] == 1  # only the first loss counted
    assert dyn.key_death[GameAction.ACTION3.value] == 0
    # A loss earlier in an attempt is an ordinary loss.
    dyn.observe(GameAction.ACTION5, None, objs, g, None, GAME_OVER)
    assert not dyn.last_death_budget and dyn.key_death[GameAction.ACTION5.value] == 2
    # The same fatal move at the same length twice is the move, not a budget.
    other = Dynamics()
    for _ in range(2):
        for _ in range(4):
            other.observe(GameAction.ACTION1, None, objs, g, extract(g), OK)
        other.observe(GameAction.ACTION5, None, objs, g, None, GAME_OVER)
    assert other.budget() is None and not other.last_death_budget
    # A longer attempt refutes the budget until losses pile up at the new length.
    dyn.new_attempt()
    for _ in range(6):
        dyn.observe(GameAction.ACTION1, None, objs, g, extract(g), OK)
    assert dyn.budget() is None
    assert "move budget" not in "\n".join(dyn.describe())


def test_contact_loss_names_the_kind_that_kills():
    """The frame of a lost game still shows the scene: a kind that stands where the
    avatar went, far more often than it is beside the avatar otherwise, kills."""
    dyn = Dynamics()

    def scene(ax, ex=None):
        g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
        g[20:22, ax : ax + 2] = 9  # avatar
        g[40:42, 40:42] = 6  # a harmless twin colour far away
        if ex is not None:
            g[20:22, ex : ex + 2] = 6  # the patrol
        return g

    for _ in range(2):
        x = 2
        for _ in range(10):  # the avatar walks right, the patrol far away
            before, after = scene(x, 30), scene(x + 2, 30)
            dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(after), OK)
            x += 2
    for _ in range(2):  # the patrol steps onto the avatar's destination: lost
        before = scene(14, 18)
        after = scene(14, 16)
        after[20:22, 14:16] = 0  # the avatar is gone
        dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(after), GAME_OVER)
    patrol = next(o.sig for o in extract(scene(2, 30)) if o.color == 6)
    assert dyn.hazard_sigs() == {patrol}
    assert "kills on contact" in "\n".join(dyn.describe())
    grid = scene(14, 18)  # the patrol within a step of the destination
    assert dyn.predict(GameAction.ACTION1, None, extract(grid), grid).verdict == "danger"
    grid = scene(2, 30)
    assert dyn.predict(GameAction.ACTION1, None, extract(grid), grid).verdict == "move"
    zone = dyn.hazard_zone(extract(grid), grid.shape)
    assert zone is not None and zone[20, 28] and not zone[20, 4]


def test_body_includes_sprites_that_move_in_lockstep_with_the_avatar():
    """A bar riding on top of the avatar is part of its body: what the bar would hit
    stops the avatar, and the body's footprint is what the planner moves."""
    dyn = Dynamics()

    def scene(ax):
        g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
        g[20:22, ax : ax + 2] = 9  # avatar 2x2
        g[19, ax : ax + 2] = 6  # a bar riding just above it
        g[19, 38:40] = 3  # a low wall: only the bar's row
        g[50:52, 50:52] = 6  # an unrelated twin colour that never moves
        return g

    x = 4
    for _ in range(8):
        before, after = scene(x), scene(x + 2)
        dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(after), OK)
        x += 2
    bar = next(o.sig for o in extract(scene(4)) if o.color == 6 and o.w == 2)
    assert dyn.avatar()[0] == 9 and dyn.companions() == {bar}
    grid = scene(20)
    objs = extract(grid)
    me = dyn.body(find(objs, dyn.avatar()), objs)
    assert me is not None and (me.x1, me.y1, me.w, me.h) == (20, 19, 2, 3) and len(me.pixels) == 6
    for _ in range(3):  # the bar hits the low wall: the whole body stops
        before = scene(36)
        dyn.observe(GameAction.ACTION1, None, extract(before), before, extract(before), OK)
    assert dyn.cell[3]["blocked"] == 3
    grid = scene(36)
    assert dyn.predict(GameAction.ACTION1, None, extract(grid), grid).verdict == "blocked"
    assert "moves with it as one body: colour-6 2x1" in "\n".join(dyn.describe())
