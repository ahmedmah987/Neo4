"""Objects on screen and how they change between two frames.

The rule-induction layer does not reason about pixels but about *objects*:
single-colour connected components with a signature (colour, size and, for
small sprites, the exact shape). ``diff`` matches the objects of two
consecutive frames and reports what moved, vanished, appeared or changed
colour. Large components (backgrounds, walls) are kept as objects so a click
on them has a signature, but they are treated as static scenery: an avatar
walking across a wall region changes that region's pixel set, and that must
not read as the wall moving.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import frame as fp

SMALL_AREA = 64  # objects up to this many pixels are matched on their exact shape
SCENERY_AREA = 400  # components at least this large are static scenery
MAX_MOVE = 24  # the furthest an object may travel in one step and still be "the same"


@dataclass(slots=True)
class Obj:
    color: int
    x1: int
    y1: int
    w: int
    h: int
    area: int
    sig: tuple  # (colour, height, width, shape-hash or None)
    pixels: np.ndarray

    @property
    def x2(self) -> int:
        return self.x1 + self.w - 1

    @property
    def y2(self) -> int:
        return self.y1 + self.h - 1

    @property
    def scenery(self) -> bool:
        return self.area >= SCENERY_AREA

    @property
    def shape_key(self) -> tuple:
        return self.sig[1:]

    def contains(self, x: int, y: int) -> bool:
        if not (self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2):
            return False
        return bool(((self.pixels[:, 0] == y) & (self.pixels[:, 1] == x)).any())

    def label(self) -> str:
        kind = "region" if self.scenery else "sprite"
        return f"colour-{self.color} {self.w}x{self.h} {kind}"


def _shape_hash(c: fp.Component) -> str:
    local = np.zeros((c.height, c.width), dtype=bool)
    local[c.pixels[:, 0] - c.y1, c.pixels[:, 1] - c.x1] = True
    return hashlib.blake2b(np.packbits(local).tobytes(), digest_size=6).hexdigest()


def extract(grid: np.ndarray, mask: np.ndarray | None = None) -> list[Obj]:
    """Objects of a frame; status-bar pixels (``mask``) are left out."""
    comps = fp.segment(fp.apply_mask(grid, mask))
    objs: list[Obj] = []
    for c in comps:
        if c.color == fp.MASK_COLOR:
            continue
        shape = _shape_hash(c) if c.area <= SMALL_AREA else None
        objs.append(
            Obj(
                color=c.color,
                x1=c.x1,
                y1=c.y1,
                w=c.width,
                h=c.height,
                area=c.area,
                sig=(c.color, c.height, c.width, shape),
                pixels=c.pixels,
            )
        )
    return objs


@dataclass(slots=True)
class Diff:
    moved: list[tuple[Obj, Obj, int, int]]  # before, after, dx, dy
    vanished: list[Obj]
    appeared: list[Obj]
    recolored: list[tuple[Obj, Obj]]
    unchanged: list[Obj]

    @property
    def changed(self) -> bool:
        return bool(self.moved or self.vanished or self.appeared or self.recolored)


def diff(before: list[Obj], after: list[Obj]) -> Diff:
    """Match objects across two frames. Scenery only ever counts as unchanged."""
    b_free = [o for o in before if not o.scenery]
    a_free = [o for o in after if not o.scenery]
    unchanged = [o for o in before if o.scenery]

    # 1. identical object at the same place
    a_index: dict[tuple, list[Obj]] = {}
    for o in a_free:
        a_index.setdefault((o.sig, o.x1, o.y1), []).append(o)
    rest_b: list[Obj] = []
    for o in b_free:
        bucket = a_index.get((o.sig, o.x1, o.y1))
        if bucket:
            unchanged.append(o)
            a_free.remove(bucket.pop())
        else:
            rest_b.append(o)

    # 2. same object elsewhere: greedy nearest match per signature
    pairs = [
        (abs(a.x1 - b.x1) + abs(a.y1 - b.y1), i, j)
        for i, b in enumerate(rest_b)
        for j, a in enumerate(a_free)
        if a.sig == b.sig and abs(a.x1 - b.x1) + abs(a.y1 - b.y1) <= MAX_MOVE
    ]
    pairs.sort()
    used_b: set[int] = set()
    used_a: set[int] = set()
    moved: list[tuple[Obj, Obj, int, int]] = []
    for _, i, j in pairs:
        if i in used_b or j in used_a:
            continue
        used_b.add(i)
        used_a.add(j)
        b, a = rest_b[i], a_free[j]
        moved.append((b, a, a.x1 - b.x1, a.y1 - b.y1))
    rest_b = [b for i, b in enumerate(rest_b) if i not in used_b]
    a_free = [a for j, a in enumerate(a_free) if j not in used_a]

    # 3. same shape at the same place in another colour
    recolored: list[tuple[Obj, Obj]] = []
    still_b: list[Obj] = []
    for b in rest_b:
        hit = next(
            (a for a in a_free if a.x1 == b.x1 and a.y1 == b.y1 and a.shape_key == b.shape_key),
            None,
        )
        if hit is not None:
            recolored.append((b, hit))
            a_free.remove(hit)
        else:
            still_b.append(b)

    return Diff(moved=moved, vanished=still_b, appeared=a_free, recolored=recolored, unchanged=unchanged)


def find(objs: list[Obj], sig: tuple) -> Obj | None:
    return next((o for o in objs if o.sig == sig), None)


def at(objs: list[Obj], x: int, y: int) -> Obj | None:
    """The object whose pixels include (x, y); smallest first so sprites win over scenery."""
    hits = [o for o in objs if o.contains(x, y)]
    return min(hits, key=lambda o: o.area) if hits else None
