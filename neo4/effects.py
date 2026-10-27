"""What a click does, learned as a template, so that clicks can be tried in imagination (Phase 60).

The explorer knows whether clicking a kind of object changed anything; to plan with clicks it needs
to know what changed. Every click is recorded as the cells it changed, with their colours before and
after, in two frames of reference:

* relative to the clicked object (offsets from its top-left corner): a tile that toggles, a piece
  that paints its neighbours;
* absolute (screen positions): a button that acts on a board elsewhere.

Objects are grouped by size and shape, not colour, since a click often recolours the object itself.
A shape's effect is *known* when its clicks agree in one frame of reference: the same cells change,
and each cell's colour changes the same way (old colour -> new colour) whenever it was seen. A known
effect predicts the frame after a click; each prediction is checked against the frame that follows,
and a shape whose prediction fails is unknown again until its clicks agree anew.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .objects import Obj

MIN_AGREE = 2  # clicks that must agree before an effect is used
MAX_CELLS = 400  # a click changing more cells than this is not modelled (a level change, a scroll)


def shape_of(o: Obj) -> tuple:
    return tuple(o.esig[1:])  # height, width, exact shape: the colour left out


@dataclass
class Effect:
    """One frame of reference's template: per cell (offset or position), colour -> colour."""

    cells: dict[tuple[int, int], dict[int, int]] = field(default_factory=dict)
    agree: int = 0

    def merge(self, changes: dict[tuple[int, int], tuple[int, int]]) -> bool:
        """Fold one click's changes in; False when they contradict the template."""
        if self.agree and set(changes) != set(self.cells):
            return False
        for cell, (old, new) in changes.items():
            m = self.cells.setdefault(cell, {})
            if m.get(old, new) != new:
                return False
            m[old] = new
        self.agree += 1
        return True


@dataclass
class ShapeModel:
    relative: Effect = field(default_factory=Effect)
    absolute: Effect = field(default_factory=Effect)
    rel_broken: bool = False
    abs_broken: bool = False
    clicks: int = 0
    checked: int = 0
    correct: int = 0

    def frame(self) -> str | None:
        """The frame of reference that explains the clicks so far, if one does."""
        if not self.rel_broken and self.relative.agree >= MIN_AGREE:
            return "relative"
        if not self.abs_broken and self.absolute.agree >= MIN_AGREE:
            return "absolute"
        return None


@dataclass
class EffectModel:
    shapes: dict[tuple, ShapeModel] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)

    @staticmethod
    def _changes(before: np.ndarray, after: np.ndarray, mask: np.ndarray | None) -> dict[tuple[int, int], tuple[int, int]] | None:
        diff = before != after
        if mask is not None:
            diff &= ~mask
        ys, xs = np.nonzero(diff)
        if len(ys) > MAX_CELLS:
            return None
        return {(int(y), int(x)): (int(before[y, x]), int(after[y, x])) for y, x in zip(ys, xs)}

    def observe(self, target: Obj, before: np.ndarray, after: np.ndarray, mask: np.ndarray | None) -> None:
        """A click on ``target`` turned ``before`` into ``after``: check the prediction, then learn."""
        changes = self._changes(before, after, mask)
        key = shape_of(target)
        m = self.shapes.setdefault(key, ShapeModel())
        if m.frame() is not None:
            predicted = self.predict(target, before, mask)
            m.checked += 1
            self.stats["checked"] += 1
            if predicted is not None and changes is not None and np.array_equal(
                    np.where(mask, before, predicted) if mask is not None else predicted,
                    np.where(mask, before, after) if mask is not None else after):
                m.correct += 1
                self.stats["correct"] += 1
        if changes is None:
            m.rel_broken = m.abs_broken = True
            return
        m.clicks += 1
        rel = {(y - target.y1, x - target.x1): c for (y, x), c in changes.items()}
        if not m.rel_broken and not m.relative.merge(rel):
            m.rel_broken = True
        if not m.abs_broken and not m.absolute.merge(changes):
            m.abs_broken = True
        if m.rel_broken and m.abs_broken and changes:
            # neither frame explains every click so far: start again from this one, so that a shape
            # whose first clicks were special (a selection, a first move) can still be learned
            m.relative, m.absolute = Effect(), Effect()
            m.relative.merge(rel)
            m.absolute.merge(changes)
            m.rel_broken = m.abs_broken = False
            self.stats["restarts"] += 1

    def known_count(self) -> int:
        """Shapes whose clicks have a known effect that changes something."""
        n = 0
        for m in self.shapes.values():
            f = m.frame()
            if f is not None and (m.relative.cells if f == "relative" else m.absolute.cells):
                n += 1
        return n

    def known(self, target: Obj) -> bool:
        m = self.shapes.get(shape_of(target))
        return m is not None and m.frame() is not None

    def predict(self, target: Obj, grid: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        """The frame after clicking ``target``, or None when the effect is unknown or a cell's colour
        was never seen at that place."""
        m = self.shapes.get(shape_of(target))
        if m is None:
            return None
        kind = m.frame()
        if kind is None:
            return None
        out = grid.copy()
        h, w = grid.shape
        effect = m.relative if kind == "relative" else m.absolute
        for (dy, dx), mapping in effect.cells.items():
            y, x = (target.y1 + dy, target.x1 + dx) if kind == "relative" else (dy, dx)
            if not (0 <= y < h and 0 <= x < w):
                return None
            old = int(grid[y, x])
            if old not in mapping:
                return None
            out[y, x] = mapping[old]
        return out


def imagine(grid: np.ndarray, mask: np.ndarray | None, model: EffectModel, extract, measure,
            max_nodes: int = 3000, max_depth: int = 12) -> list[Obj] | None:
    """A shortest-first, closest-first search in imagination: clicks on objects whose shape has a known
    effect, each frame predicted by the model, until ``measure(objs, grid)`` reaches 0. Returns the
    objects to click, in order (as they stand when clicked), or None."""
    import heapq

    start_objs = extract(grid, mask)
    h0 = measure(start_objs, grid)
    if h0 is None or h0 == 0:
        return None
    seen = {grid.tobytes()}
    frontier: list = [(h0, 0, 0, grid, start_objs, [])]
    counter = 1
    while frontier and counter < max_nodes:
        h, depth, _, g, objs, path = heapq.heappop(frontier)
        if depth >= max_depth:
            continue
        for o in objs:
            if o.scenery or not model.known(o):
                continue
            nxt = model.predict(o, g, mask)
            if nxt is None:
                continue
            key = nxt.tobytes()
            if key in seen:
                continue
            seen.add(key)
            nobjs = extract(nxt, mask)
            nh = measure(nobjs, nxt)
            if nh is None:
                continue
            if nh == 0:
                return path + [o]
            counter += 1
            heapq.heappush(frontier, (nh, depth + 1, counter, nxt, nobjs, path + [o]))
    return None
