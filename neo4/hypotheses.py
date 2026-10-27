"""Goal hypotheses read from the screen, and a memory of what each action does to them (Phase 59).

A person facing a new level guesses its goal from the screen and tests the guess; the explorer
instead tries actions until a level happens to end. This module supplies guesses of the kinds a
person makes first, each with a measure of how far the screen is from it, so that the agent can
learn which actions move which measure and steer by the ones that move:

* ``reach``: the player is to reach a kind seen once (an exit, a door, a marked cell);
  measured by the distance between the player and it;
* ``clear``: every object of a kind is to go (collected, removed, matched away); measured by
  how many are left;
* ``match``: two panels of the same size are to become the same (a reference and a board);
  measured by the number of cells in which they differ;
* ``unify``: objects of one shape are to take one colour; measured by how many colours they show.

Nothing here is specific to a public game: the guesses come from the structure of the first frame
of a level (rarity, repetition, equal sizes), and none is kept that no action has ever moved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .objects import Obj

MAX_CLEAR_KINDS = 6  # kinds with several instances proposed as "clear", the most numerous first
MAX_REACH = 6  # single-instance kinds proposed as "reach", the smallest first
MIN_PANEL = 3  # a panel's smaller side, in cells
MAX_PANELS = 12


@dataclass(slots=True)
class Hypothesis:
    """A guess at a level's goal: a kind, its parameters, and the measure's value when met."""

    kind: str  # reach | clear | match | unify
    params: tuple
    goal: int = 0
    first: int | None = None  # the measure when the level began
    best: int | None = None  # the lowest value seen in this level

    @property
    def key(self) -> tuple:
        return (self.kind, self.params)


def background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def panels(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Rectangles (y1, x1, h, w) of the connected regions of non-background cells, any colour,
    at least MIN_PANEL on each side: the boards, reference pictures and frames of a screen."""
    bg = background(grid)
    mask = grid != bg
    seen = np.zeros_like(mask)
    out = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack, ys, xs = [(y, x)], [], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            y1, x1, y2, x2 = min(ys), min(xs), max(ys), max(xs)
            if y2 - y1 + 1 >= MIN_PANEL and x2 - x1 + 1 >= MIN_PANEL:
                out.append((y1, x1, y2 - y1 + 1, x2 - x1 + 1))
    out.sort(key=lambda r: (-(r[2] * r[3]), r))
    return out[:MAX_PANELS]


def propose(objs: list[Obj], grid: np.ndarray, avatar_sig: tuple | None) -> list[Hypothesis]:
    """Guesses from a level's first frame."""
    free = [o for o in objs if not o.scenery]
    counts = Counter(o.sig for o in free)
    out: list[Hypothesis] = []
    singles = sorted((o for o in free if counts[o.sig] == 1 and o.sig != avatar_sig),
                     key=lambda o: (o.area, o.y1, o.x1))
    for o in singles[:MAX_REACH]:
        out.append(Hypothesis("reach", (o.sig,)))
    many = sorted((s for s, n in counts.items() if 2 <= n <= 40 and s != avatar_sig),
                  key=lambda s: (-counts[s], repr(s)))
    for s in many[:MAX_CLEAR_KINDS]:
        out.append(Hypothesis("clear", (s,)))
    shapes = Counter(o.sig[1:] for o in free)
    for shape, n in sorted(shapes.items(), key=lambda kv: (-kv[1], repr(kv[0]))):
        if n >= 3 and len({o.color for o in free if o.sig[1:] == shape}) >= 2:
            out.append(Hypothesis("unify", (shape,), goal=1))
    rects = panels(grid)
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            if a[2:] == b[2:] and not np.array_equal(_cut(grid, a), _cut(grid, b)):
                out.append(Hypothesis("match", (a, b)))
    return out


def _cut(grid: np.ndarray, r: tuple[int, int, int, int]) -> np.ndarray:
    y, x, h, w = r
    return grid[y:y + h, x:x + w]


def measure(h: Hypothesis, objs: list[Obj], grid: np.ndarray, me: Obj | None) -> int | None:
    """How far the screen is from the hypothesis (``h.goal`` when met); None when unmeasurable."""
    if h.kind == "reach":
        if me is None:
            return None
        targets = [o for o in objs if o.sig == h.params[0]]
        if not targets:
            return 0  # entered or taken
        return min(_gap(me, t) for t in targets)
    if h.kind == "clear":
        return sum(1 for o in objs if o.sig == h.params[0])
    if h.kind == "unify":
        colours = {o.color for o in objs if not o.scenery and o.sig[1:] == h.params[0]}
        return max(1, len(colours))
    if h.kind == "match":
        a, b = h.params
        return int((_cut(grid, a) != _cut(grid, b)).sum())
    return None


def _gap(a: Obj, b: Obj) -> int:
    dx = max(0, max(a.x1, b.x1) - min(a.x2, b.x2) - 1)
    dy = max(0, max(a.y1, b.y1) - min(a.y2, b.y2) - 1)
    return dx + dy


@dataclass
class CauseMemory:
    """For each action class, how often it moved each hypothesis's measure, and which way."""

    moves: dict[tuple, Counter] = field(default_factory=dict)  # (hyp key, action class) -> Counter

    def record(self, hyp_key: tuple, action_class: object, before: int | None, after: int | None) -> None:
        if before is None or after is None:
            return
        c = self.moves.setdefault((hyp_key, action_class), Counter())
        c["n"] += 1
        if after < before:
            c["closer"] += 1
            c["gain"] += before - after
        elif after > before:
            c["farther"] += 1
            c["loss"] += after - before

    def live(self, hyp_key: tuple) -> bool:
        """Some action has moved this measure at least once."""
        return any(k[0] == hyp_key and (c["closer"] or c["farther"]) for k, c in self.moves.items())

    def helped(self, hyp_key: tuple) -> bool:
        """Some action has moved this measure closer to its goal."""
        return any(k[0] == hyp_key and c["closer"] for k, c in self.moves.items())

    def mean_gain(self, hyp_key: tuple, action_class: object) -> float:
        c = self.moves.get((hyp_key, action_class))
        if not c or not c["n"]:
            return 0.0
        return (c["gain"] - c["loss"]) / c["n"]

    def best_actions(self, hyp_key: tuple) -> list[tuple[float, object]]:
        """Action classes by mean progress on this measure, most helpful first."""
        out = []
        for (hk, cls), c in self.moves.items():
            if hk == hyp_key and c["n"]:
                out.append(((c["gain"] - c["loss"]) / c["n"], cls))
        out.sort(key=lambda t: (-t[0], repr(t[1])))
        return out
