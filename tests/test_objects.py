import numpy as np

from neo4 import frame as fp
from neo4.objects import at, diff, extract, find


def scene(avatar_x: int = 10, *, extra=None) -> np.ndarray:
    g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
    g[0:64, 0:2] = 3  # a wall strip: scenery-sized region
    g[20:23, avatar_x : avatar_x + 3] = 9  # 3x3 avatar
    g[21, avatar_x + 1] = 0  # hole, so the shape is not a plain square
    g[40:42, 50:52] = 12  # 2x2 item
    if extra:
        extra(g)
    return g


def test_extract_signatures_and_scenery():
    objs = extract(scene())
    bg = max(objs, key=lambda o: o.area)
    assert bg.scenery and bg.sig[3] is None
    avatar = next(o for o in objs if o.color == 9)
    assert avatar.sig[:3] == (9, 3, 3) and avatar.sig[3] is not None
    assert avatar.area == 8 and not avatar.scenery
    assert at(objs, 11, 20) is avatar
    hole = at(objs, 11, 21)  # enclosed by the avatar, so a separate 1x1 object
    assert hole is not bg and hole.area == 1 and hole.color == 0
    assert at(objs, 5, 5) is bg


def test_diff_detects_a_move_and_ignores_background_reshaping():
    before, after = extract(scene(10)), extract(scene(13))
    d = diff(before, after)
    assert {b.color for b, _, _, _ in d.moved} == {9, 0}  # the avatar and the hole inside it
    assert all((dx, dy) == (3, 0) for _, _, dx, dy in d.moved)
    assert not d.vanished and not d.appeared and not d.recolored
    assert any(o.scenery for o in d.unchanged)


def test_diff_vanish_appear_recolor():
    before = extract(scene())
    after = extract(scene(extra=lambda g: (g.__setitem__((slice(40, 42), slice(50, 52)), 0),
                                           g.__setitem__((slice(5, 7), slice(30, 32)), 7))))
    d = diff(before, after)
    assert [o.color for o in d.vanished] == [12]
    assert [o.color for o in d.appeared] == [7]
    recol = extract(scene(extra=lambda g: g.__setitem__((slice(40, 42), slice(50, 52)), 5)))
    d = diff(before, recol)
    assert len(d.recolored) == 1 and d.recolored[0][0].color == 12 and d.recolored[0][1].color == 5
    assert d.changed


def test_find_and_unchanged():
    objs = extract(scene())
    avatar = next(o for o in objs if o.color == 9)
    assert find(objs, avatar.sig) is avatar
    d = diff(objs, extract(scene()))
    assert not d.changed and len(d.unchanged) == len(objs)
