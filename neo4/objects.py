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
    sig: tuple  # (colour, smaller side, larger side, orientation-free shape hash or None)
    esig: tuple  # (colour, height, width, exact shape hash or None): orientation matters
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


def _exact_hash(c: fp.Component) -> str:
    local = np.zeros((c.height, c.width), dtype=bool)
    local[c.pixels[:, 0] - c.y1, c.pixels[:, 1] - c.x1] = True
    return hashlib.blake2b(np.packbits(local).tobytes(), digest_size=6).hexdigest()


def _shape_hash(c: fp.Component) -> str:
    """Hash of the sprite's shape, the same under rotation and mirroring.

    A sprite with an orientation marker (tu93's player has an "eye" pixel
    that turns with it) must keep one identity as it turns, so the hash is
    taken over the canonical member of the eight dihedral transforms.
    """
    local = np.zeros((c.height, c.width), dtype=bool)
    local[c.pixels[:, 0] - c.y1, c.pixels[:, 1] - c.x1] = True
    variants = []
    for k in range(4):
        r = np.rot90(local, k)
        variants.append(r)
        variants.append(np.fliplr(r))
    canon = min(variants, key=lambda m: (m.shape, np.packbits(m).tobytes()))
    return hashlib.blake2b(np.packbits(canon).tobytes() + bytes(canon.shape), digest_size=6).hexdigest()


class Objects(list):
    """The objects of one frame, with a pixel -> object lookup built on demand."""

    shape: tuple[int, int] = (fp.GRID, fp.GRID)
    _index: np.ndarray | None = None

    def index(self) -> np.ndarray:
        if self._index is None:
            idx = np.full(self.shape, -1, dtype=np.int32)
            # paint big objects first so that small ones win where they overlap
            for i in sorted(range(len(self)), key=lambda i: -self[i].area):
                o = self[i]
                idx[o.pixels[:, 0], o.pixels[:, 1]] = i
            self._index = idx
        return self._index


def extract(grid: np.ndarray, mask: np.ndarray | None = None) -> Objects:
    """Objects of a frame; status-bar pixels (``mask``) are left out."""
    comps = fp.segment(fp.apply_mask(grid, mask))
    objs = Objects()
    objs.shape = grid.shape
    for c in comps:
        if c.color == fp.MASK_COLOR:
            continue
        small = c.area <= SMALL_AREA
        shape = _shape_hash(c) if small else None
        # `sig` is orientation-free (an avatar keeps its identity as it turns);
        # `esig` is exact (a click target's orientation can be what it means).
        objs.append(
            Obj(
                color=c.color,
                x1=c.x1,
                y1=c.y1,
                w=c.width,
                h=c.height,
                area=c.area,
                sig=(c.color, min(c.height, c.width), max(c.height, c.width), shape),
                esig=(c.color, c.height, c.width, _exact_hash(c) if small else None),
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


def erase(grid: np.ndarray, objs: list[Obj], sigs: frozenset | set) -> np.ndarray:
    """Copy of ``grid`` with the objects of the given signatures painted over.

    Each object is filled with the most common colour of the cells around it,
    so that an object drifting across a uniform background leaves no trace.
    """
    out = grid.copy()
    h, w = grid.shape
    for o in objs:
        if o.sig not in sigs:
            continue
        ys, xs = o.pixels[:, 0], o.pixels[:, 1]
        ring: list[int] = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            ring.extend(grid[ny[ok], nx[ok]].tolist())
        own = set(zip(ys.tolist(), xs.tolist()))
        ring = [c for c, (y, x) in zip(ring, _ring_coords(ys, xs, h, w)) if (y, x) not in own]
        fill = max(set(ring), key=ring.count) if ring else 0
        out[ys, xs] = fill
    return out


def _ring_coords(ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        coords.extend(zip(ny[ok].tolist(), nx[ok].tolist()))
    return coords


def find(objs: list[Obj], sig: tuple) -> Obj | None:
    return next((o for o in objs if o.sig == sig), None)


def at(objs: list[Obj], x: int, y: int) -> Obj | None:
    """The object whose pixels include (x, y); smallest first so sprites win over scenery."""
    if isinstance(objs, Objects):
        idx = objs.index()
        if not (0 <= y < idx.shape[0] and 0 <= x < idx.shape[1]):
            return None
        i = int(idx[y, x])
        return objs[i] if i >= 0 else None
    hits = [o for o in objs if o.contains(x, y)]
    return min(hits, key=lambda o: o.area) if hits else None
