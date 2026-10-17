import numpy as np

from neo4 import frame as fp


def blank() -> np.ndarray:
    return np.zeros((fp.GRID, fp.GRID), dtype=np.int8)


def test_segment_finds_components_with_bboxes():
    f = blank()
    f[5:15, 5:15] = 9  # 10x10 square
    f[20, 0:30] = 12  # 1x30 line
    comps = fp.segment(f)
    by_color = {c.color: c for c in comps}
    assert set(by_color) == {0, 9, 12}
    sq = by_color[9]
    assert (sq.x1, sq.y1, sq.x2, sq.y2) == (5, 5, 14, 14)
    assert sq.area == 100 and sq.is_rectangle
    assert by_color[12].width == 30 and by_color[12].height == 1
    assert sum(c.area for c in comps) == fp.GRID * fp.GRID


def test_segment_uses_4_connectivity():
    f = blank()
    f[0, 0] = 7
    f[1, 1] = 7  # diagonal neighbour only
    comps = [c for c in fp.segment(f) if c.color == 7]
    assert len(comps) == 2


def test_click_point_lies_inside_component():
    f = blank()
    f[10:14, 10:30] = 8
    f[12, 15:25] = 0  # hole through the middle
    c = next(c for c in fp.segment(f) if c.color == 8)
    x, y = c.click_point()
    assert f[y, x] == 8


def test_status_bar_long_bar_at_bottom_edge():
    f = blank()
    f[63, 2:60] = 12
    comps = fp.segment(f)
    mask = fp.status_bar_mask(comps)
    assert mask[63, 2:60].all()
    assert mask.sum() == 58


def test_status_bar_dot_style_at_top_edge():
    f = blank()
    for x in (2, 6, 10):  # three identical 2x2 dots hugging the top edge
        f[0:2, x : x + 2] = 7
    f[30:32, 30:32] = 7  # a twin away from the edge must not count or be masked
    mask = fp.status_bar_mask(fp.segment(f))
    assert mask[0:2, 2:4].all() and mask[0:2, 6:8].all() and mask[0:2, 10:12].all()
    assert not mask[30:32, 30:32].any()
    assert mask.sum() == 12


def test_two_dots_are_not_a_bar():
    f = blank()
    f[0:2, 2:4] = 7
    f[0:2, 6:8] = 7
    assert fp.status_bar_mask(fp.segment(f)).sum() == 0


def test_interior_square_is_not_a_bar():
    f = blank()
    f[20:30, 20:60] = 5  # wide but not at an edge
    assert fp.status_bar_mask(fp.segment(f)).sum() == 0


def test_state_key_ignores_masked_pixels_only():
    f = blank()
    f[63, 2:60] = 12
    f[10:12, 10:12] = 9
    mask = fp.status_bar_mask(fp.segment(f))
    k0 = fp.state_key(f, mask)
    g = f.copy()
    g[63, 40:60] = 0  # bar shrinks
    assert fp.state_key(g, mask) == k0
    h = f.copy()
    h[10:12, 10:12] = 0
    h[12:14, 12:14] = 9  # object moves
    assert fp.state_key(h, mask) != k0
    assert fp.state_key(f, None) != fp.state_key(g, None)


def test_click_groups():
    f = blank()
    f[5:15, 5:15] = 9  # salient, medium -> 0
    f[20:30, 20:30] = 2  # dull, medium -> 1
    f[40, 0:40] = 11  # salient, thin -> 2
    comps = fp.segment(fp.apply_mask(f, None))
    groups = {c.color: g for c, g in zip(comps, fp.click_groups(comps))}
    assert groups[9] == 0 and groups[2] == 1 and groups[11] == 2 and groups[0] == 3
    masked = fp.apply_mask(f, f == 11)
    comps = fp.segment(masked)
    assert max(fp.click_groups(comps)) == 4


def test_segment_without_scipy_matches_scipy(monkeypatch):
    f = blank()
    f[5:15, 5:15] = 9
    f[20, 0:30] = 12
    f[0, 0] = 7
    f[1, 1] = 7
    with_scipy = [(c.color, c.x1, c.y1, c.x2, c.y2, c.area) for c in fp.segment(f)]
    monkeypatch.setattr(fp, "ndimage", None)
    without = [(c.color, c.x1, c.y1, c.x2, c.y2, c.area) for c in fp.segment(f)]
    assert sorted(with_scipy) == sorted(without)
