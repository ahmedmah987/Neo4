"""The pointer family: select a handle, then click where it should go.

Some puzzles move things by pointing: clicking a handle selects it, clicking a
free place sends the selected handle there, and what has to reach a goal is a
body tied to its handles (it sits at their centroid). Nothing about a click's
target says where the handle goes; the click's coordinates do. This solver
learns that rule and the ties from the frames and plans the clicks directly.

From the frames alone:

* a click that changes the frame and after which a component's centre is exactly
  the clicked point *sent* that component there; two such clicks for components
  of one kind (colour-free shape) make the kind a *handle* kind, and the colour
  the sent handle had is the colour of the selected handle;
* a click on a handle of that kind selects it;
* a *thing* is a set of touching components of any colours (handles, the
  background and the thin lines drawn between things left out); a *body* is a
  thing whose centre has been, in every frame it was seen in, the mean of the
  centres of the same set of handles. Once tied, a body is known by its handles,
  so it is still the same body when its colours change;
* a body's goal is a cluster of small still things in the level's first frame
  whose colours are all among the body's: of those, the one that leaves the
  fewest of the body's colours unmatched and matches the most. When a level
  ends, where each body stood relative to its goal is kept for the next levels;
  otherwise any place where the body covers the goal's centre is tried, nearest
  first;
* a still thing that could be a body but is tied to nothing may give a body its
  colours when the body lands on it (it is tried once; the frames say whether it
  happened). A body that has no goal yet goes to the things whose colours a goal
  needs, and adds none it does not, before going to that goal.

To put a body on a target with ``k`` handles it moves one handle to ``k`` times
the target minus the other handles' centres, selecting it first if needed; if no
handle can go there (off screen, onto something still, onto another handle), it
moves one handle somewhere free first and then another, or, for three handles or
more, gathers them round the target. Places a handle could not go and places the
body bounced from are not tried again.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations, permutations
from math import cos, hypot, pi, sin

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp
from .marker import Comp, components, cover_places, centre

Action = tuple[GameAction, dict[str, int] | None]

GOTO_PROOF = 2  # clicks that sent a handle to the clicked point before the rule is trusted
MAX_HANDLES = 8  # handles a body may be tied to are searched among this many
MAX_TIE = 4  # a body is tied to at most this many handles
MAX_LEVEL_ACTIONS = 120  # actions the solver may spend on one level
UNPROVEN_ACTIONS = 40  # ... in a game it has not completed a level of
MAX_FAILS = 8  # clicks that did not do what was planned, per level
UNPOINTED_CLICKS = 200  # clicks after which a game with no goto is not a pointer game
LATTICE = 2  # spacing of the places tried for a handle moved out of the way
MARK_SIZE = 12  # a goal's parts fit in this many pixels square
GOAL_GAP = 4  # parts of a goal of one colour this close together form one goal
MAX_BODY = 16  # a body fits in this many pixels square
MAX_POTS = 3  # things a body may take colours from on its way to one goal


@dataclass(frozen=True)
class Obj:
    """Touching components of any colours: a thing as it is drawn."""

    x: int
    y: int
    w: int
    h: int
    rel: frozenset  # (dy, dx) pixel offsets from the top-left
    colors: frozenset
    tie: frozenset = frozenset()  # the handles it sits between, for a body

    @property
    def area(self) -> int:
        return len(self.rel)

    @property
    def sig(self) -> tuple:
        return (self.colors, self.rel)

    @property
    def place(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def pixels(self) -> frozenset:
        return frozenset((self.x + dx, self.y + dy) for dy, dx in self.rel)


def _product_upto(lists: list[list], depth: int):
    """Combinations whose largest index is exactly ``depth``."""
    def rec(i, acc, hit):
        if i == len(lists):
            if hit:
                yield tuple(acc)
            return
        for j, x in enumerate(lists[i][:depth + 1]):
            yield from rec(i + 1, acc + [x], hit or j == depth)
    yield from rec(0, [], False)


def box_centre(c) -> tuple[float, float]:
    return (c.x + (c.w - 1) / 2.0, c.y + (c.h - 1) / 2.0)


def _near(c, m: tuple[float, float]) -> bool:
    p = box_centre(c)
    return abs(p[0] - m[0]) < 1 and abs(p[1] - m[1]) < 1


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _grown(px) -> set:
    """Pixels and their four neighbours."""
    return {(x + dx, y + dy) for x, y in px for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))}


def _disjoint(sets) -> bool:
    seen: set = set()
    for s in sets:
        if seen & s:
            return False
        seen |= s
    return True


def _clusters(comps: list[Comp], gap: int) -> list[list[Comp]]:
    """Components close together (union-find): of one colour within ``gap``
    pixels, of different colours a pixel apart at most (two goals side by side
    stay two)."""
    parent = list(range(len(comps)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(comps):
        for j in range(i + 1, len(comps)):
            b = comps[j]
            gx = max(b.x - (a.x + a.w - 1), a.x - (b.x + b.w - 1), 0) - 1
            gy = max(b.y - (a.y + a.h - 1), a.y - (b.y + b.h - 1), 0) - 1
            if max(gx, gy) <= (gap if a.color == b.color else 1):
                parent[root(i)] = root(j)
    out: dict[int, list[Comp]] = {}
    for i, c in enumerate(comps):
        out.setdefault(root(i), []).append(c)
    return list(out.values())


class PointerSolver:
    name = "pointer"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.confirmed = False
        self.strikes = 0
        self.stats: Counter = Counter()
        self.goto_kinds: Counter = Counter()  # kind -> clicks that sent one of it to the clicked point
        self.sel_colors: Counter = Counter()  # colours the sent handles had
        # where a body stood relative to its goal when a level ended: (w, h) -> centre offset
        self.offsets: dict[tuple, tuple[float, float]] = {}
        self.moved_colors: set[int] = set()  # colours seen at a new place, over the game
        self.absorb: bool | None = None  # does a body take the colours of a still thing it lands on?
        self.pot_fails = 0  # landings on such things that took nothing
        self._tpl: Comp | None = None  # a handle as drawn: its shape, to find one drawn into something
        self.handle_colors: set[int] = set()  # colours handles have been drawn in
        self.level = -1
        self.in_control = False
        self.clicks = 0
        self._new_level(None)

    # -- per level -------------------------------------------------------------

    def _new_level(self, grid: np.ndarray | None) -> None:
        self.grid = grid
        self.mask = fp.status_bar_mask(fp.segment(grid)) if grid is not None else None
        self.bg = int(np.bincount(grid.ravel()).argmax()) if grid is not None else -1
        self.comps: list[Comp] = components(grid, self.mask) if grid is not None else []
        self.first = list(self.comps)
        self.first_grid = grid
        # frames kept until the handle kind is known, to number the handles from the start
        self.history: list[list[Comp]] = [self.comps] if grid is not None else []
        self._track_kind: str | None = None
        self._ids: dict[int, tuple[float, float]] = {}  # handle number -> centre, now
        self._cur: dict[int, Comp] = {}  # handle number -> component, now
        self._snaps = 0  # frames seen with every handle placed
        self._fits_by: dict[tuple, set] = {}  # a thing's look -> handle sets that fit every frame it was in
        self._seen_at: dict[tuple, set] = {}  # a thing's look -> places it was seen at
        self._sig_snaps: Counter = Counter()  # a thing's look -> frames it was seen in
        self._tied: dict[frozenset, Obj] = {}  # handle set -> its body, as last seen whole
        self._ambiguous: dict[tuple, list[frozenset]] = {}  # a look -> handle sets that might hold it
        self.bad_points: set = set()  # (kind, x, y): a handle sent there did not go
        self.bad_places: set = set()  # (tie, x, y): a body placed there bounced or did not end the level
        self.bad_pots: set = set()  # (look, place): a body landed there and took nothing
        self.level_actions = 0
        self.fails = 0
        self.gave_up = False
        self.expect: dict | None = None
        self._goal_cache = None
        self._first_cache = None
        self._obj_cache = None
        self._body_cache = None
        self._solid_cache = None
        self._hcache = None
        self._last_step: tuple | None = None
        self._last_key = None
        self.comps_key = None

    # -- observing --------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled:
            return
        if obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            if obs.state == GameState.GAME_OVER and self.in_control:
                self.fails += 2
            return
        if 6 not in obs.available_actions:
            self.level = -1  # nothing to point with; start afresh if clicks appear
            return
        grid = np.asarray(obs.frame[-1])
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0:
                if len(obs.frame) > 1:
                    self._learn_offsets(np.asarray(obs.frame[-2]), data)
                if self.in_control:
                    self.confirmed = True
                    self.stats["levels"] += 1
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level(grid)
            self._sync(self.comps)  # with the handle kind known, the ties from the first frame
            return
        prev = self.comps
        after = components(grid, self.mask)
        self.grid = grid
        self.comps = after
        self.comps_key = frozenset((c.color, c.shape, c.x, c.y) for c in after)  # the status bar left out
        self._obj_cache = self._body_cache = None
        if action is None or action is GameAction.RESET:
            # a fresh attempt: the handles are back where the level began; what the
            # frames taught about the ties stays
            if self._track_kind is not None:
                start = self._start_handles()
                self._ids = {i: box_centre(c) for i, c in enumerate(start)}
                self._cur = dict(enumerate(start))
            else:
                self.history = [after]
            self.expect = None
            return
        if self._track_kind is None:
            self.history.append(after)
        before_keys = {(c.color, c.shape, c.x, c.y) for c in prev}
        after_keys = {(c.color, c.shape, c.x, c.y) for c in after}
        for c in after:
            if (c.color, c.shape, c.x, c.y) not in before_keys:
                self.moved_colors.add(c.color)
        if data is not None and action.is_complex():
            self.clicks += 1
            if self.clicks >= UNPOINTED_CLICKS and self._handle_kind() is None:
                self.disabled = True  # nothing has ever gone where it was pointed
                self.history = []
                return
            pt = (int(data["x"]), int(data["y"]))
            gone = [c for c in prev if (c.color, c.shape, c.x, c.y) not in after_keys]
            new = [c for c in after if (c.color, c.shape, c.x, c.y) not in before_keys]
            gone_kinds = {c.kind for c in gone}
            for c in new:
                cx, cy = box_centre(c)
                if abs(cx - pt[0]) <= 0.5 and abs(cy - pt[1]) <= 0.5 and c.kind in gone_kinds:
                    self.goto_kinds[c.kind] += 1
                    self.sel_colors[c.color] += 1
                    self.stats["gotos"] += 1
                    break
        self._sync(after, pt if data is not None and action.is_complex() else None)
        self._check(action, data)

    def _check(self, action, data) -> None:
        """Did the planned click do what was planned?"""
        exp, self.expect = self.expect, None
        if exp is None:
            return
        key = (exp["kind"], exp.get("handle"), exp.get("at"), exp.get("to"))
        if key == self._last_step and self.comps_key == self._last_key:
            # the same step again and nothing changed: it does nothing here
            self.bad_points.add((exp.get("hkind", "select"), *(exp.get("to") or exp.get("at"))))
            self.fails += 1
            self.stats["no_change"] += 1
        self._last_step, self._last_key = key, self.comps_key
        if exp["kind"] == "select":
            if self._selected() != exp["handle"]:
                self.bad_points.add(("select", exp["at"][0], exp["at"][1]))
                self.fails += 1
                self.stats["mismatches"] += 1
            return
        hs = self._handles()
        h = hs.get(exp["handle"])
        body = exp.get("body")
        if body is not None:
            now = next((f for f in self._bodies() if (f.tie, f.x, f.y) == body), None)
            if now is not None:
                if exp.get("pot") is not None:
                    self._check_pot(exp["pot"], exp["had"], now)
                return  # the body is where it was sent
        if h is None or box_centre(h) != exp["to"]:
            # the handle did not go there (a wall), or went and came back (the body bounced)
            self.bad_points.add((exp["hkind"], exp["to"][0], exp["to"][1]))
            if exp.get("body") is not None:
                self.bad_places.add(exp["body"])
            self.fails += 1
            self.stats["mismatches"] += 1

    def _check_pot(self, pot: tuple, had: frozenset, body: Obj) -> None:
        """A body landed on a still thing to take its colours: did it? (Drawn over
        it, the body hides it; its own colours say.)"""
        sig, _ = pot
        if body.colors != had and (sig[0] - had) <= body.colors:
            self.absorb = True
            self.stats["pot_taken"] += 1
            return
        self.bad_pots.add(pot)
        self.pot_fails += 1
        if self.absorb is None and self.pot_fails >= 2:
            self.absorb = False
        self.stats["pot_kept"] += 1

    def _learn_offsets(self, final: np.ndarray, data: dict | None) -> None:
        """Where each body stood relative to its goal when the level ended. The
        winning click moved the selected handle to the clicked point."""
        if self._handle_kind() is None or not self._tied:
            return
        comps = components(final, self.mask)
        objs = [o for o in self._objects(comps) if self._body_like(o)]
        ids = dict(self._ids)
        sel = self._selected()
        if data is not None and sel is not None and sel in ids:
            ids[sel] = (int(data["x"]), int(data["y"]))
        goals = self._goals()
        for tie, body in self._tied.items():
            if any(i not in ids for i in tie):
                continue
            m = (sum(ids[i][0] for i in tie) / len(tie), sum(ids[i][1] for i in tie) / len(tie))
            seen = [o for o in objs if o.rel == body.rel and _dist(box_centre(o), m) < 2]
            if seen:
                o = min(seen, key=lambda o: _dist(box_centre(o), m))
            else:
                # drawn into something (its goal): where its handles put it
                o = replace(body, x=int(m[0] - (body.w - 1) / 2), y=int(m[1] - (body.h - 1) / 2))
            goal = self._best_goal(o.colors, box_centre(o), goals)
            if goal is None:
                continue
            fc, gc = box_centre(o), goal[0]
            if abs(gc[0] - fc[0]) + abs(gc[1] - fc[1]) <= o.w + o.h:
                self.offsets[(o.w, o.h)] = (gc[0] - fc[0], gc[1] - fc[1])
                self.stats["offsets"] += 1

    # -- the model --------------------------------------------------------------

    def _handle_kind(self) -> str | None:
        if not self.goto_kinds:
            return None
        kind, n = self.goto_kinds.most_common(1)[0]
        return kind if n >= GOTO_PROOF else None

    def _sync(self, latest: list[Comp], pt: tuple[int, int] | None = None) -> None:
        """Number the handles: from the level's first frame on, a handle keeps its
        number while it stays where it was, the one that moved takes the number left
        over. Built from the kept frames once the handle kind is known, then frame
        by frame."""
        kind = self._handle_kind()
        if kind is None:
            return
        if kind != self._track_kind:
            self._track_kind = kind
            if self._tpl is None or self._tpl.kind != kind:
                self._tpl = next((c for c in self.first + latest if c.kind == kind), None)
            self.handle_colors |= {c.color for c in self.first + latest if c.kind == kind}
            self._ids, self._cur, self._snaps = {}, {}, 0
            self._fits_by, self._seen_at, self._sig_snaps = {}, {}, Counter()
            self._tied, self._ambiguous = {}, {}
            start = self._start_handles()
            if not start:
                return
            self._ids = {i: box_centre(c) for i, c in enumerate(start)}
            self._cur = dict(enumerate(start))
            for frame in self.history:
                self._advance(frame)
            self.history = []
            return
        self._advance(latest, pt)

    def _start_handles(self) -> list[Comp]:
        """The handles as the level began; none if there are too many to be handles."""
        start = self._handles_in(self.first)
        return start if len(start) <= MAX_HANDLES else []

    def _handles_in(self, comps: list[Comp]) -> list[Comp]:
        """The handles in a frame: the components of the handle kind, and handles
        drawn into something of their colour they touch, found by their shape (its
        holes included) inside the larger component."""
        if self._hcache is not None and self._hcache[0] is comps:
            return self._hcache[1]
        kind, tpl = self._track_kind, self._tpl
        out = [c for c in comps if c.kind == kind]
        if tpl is not None and kind is not None:
            hole = {(dy, dx) for dy in range(tpl.h) for dx in range(tpl.w)} - tpl.rel
            dy0, dx0 = min(tpl.rel)
            for c in comps:
                if c.kind == kind or c.color not in self.handle_colors or c.area <= tpl.area:
                    continue
                px = {(c.x + dx, c.y + dy) for dy, dx in c.rel}
                for x, y in sorted(px, key=lambda q: (q[1], q[0])):
                    if (x, y) not in px:
                        continue
                    top = (x - dx0, y - dy0)
                    if all((top[0] + dx, top[1] + dy) in px for dy, dx in tpl.rel) and \
                            not any((top[0] + dx, top[1] + dy) in px for dy, dx in hole):
                        out.append(replace(tpl, color=c.color, x=top[0], y=top[1],
                                           point=(top[0] + tpl.point[0] - tpl.x, top[1] + tpl.point[1] - tpl.y)))
                        px -= {(top[0] + dx, top[1] + dy) for dy, dx in tpl.rel}
                        self.stats["handles_found_inside"] += 1
        if comps is self.comps or comps is self.first:
            self._hcache = (comps, out)
        return out

    def _advance(self, frame: list[Comp], pt: tuple[int, int] | None = None) -> None:
        if not self._ids:
            return
        hs = self._handles_in(frame)
        self.handle_colors |= {c.color for c in hs}
        if len(hs) > len(self._ids):
            return
        left = dict(self._ids)
        cur, rest = {}, []
        for c in hs:
            cc = box_centre(c)
            hit = next((i for i, p in left.items() if p == cc), None)
            if hit is None:
                rest.append(c)
            else:
                cur[hit] = c
                del left[hit]
        # a handle at a new place is the one that was selected, if it is unaccounted
        # for; a handle not seen (hidden under a body) keeps its last place
        sel_col = self.sel_colors.most_common(1)[0][0] if self.sel_colors else None
        was_sel = next((i for i, c in self._cur.items() if c.color == sel_col and i in left), None)
        for c in rest:
            if was_sel is not None and was_sel in left:
                i = was_sel
            else:
                i = min(left) if left else None
            if i is None:
                return
            cur[i] = c
            del left[i]
        if was_sel in left and pt is not None and self._shows(self._cur[was_sel], pt):
            # drawn into something of its colour where it was sent (a goal's flash):
            # it is there all the same
            c = self._cur[was_sel]
            x0, y0 = int(round(pt[0] - (c.w - 1) / 2)), int(round(pt[1] - (c.h - 1) / 2))
            cur[was_sel] = replace(c, x=x0, y=y0, point=pt)
            del left[was_sel]
        for i in left:
            if i in self._cur:
                cur[i] = self._cur[i]
        self._cur = cur
        self._ids = {i: box_centre(c) for i, c in cur.items()}
        if was_sel is None or was_sel not in left:
            # a hidden handle that was not selected has not moved: the frame says where
            # every handle is unless the selected one is hidden
            self._snaps += 1
            self._observe_bodies(frame)
        self._body_cache = None

    def _shows(self, c: Comp, pt: tuple[int, int]) -> bool:
        """Does the frame show most of handle ``c``'s pixels, in its colour, centred on ``pt``?"""
        x0, y0 = int(round(pt[0] - (c.w - 1) / 2)), int(round(pt[1] - (c.h - 1) / 2))
        h, w = self.grid.shape
        hits = sum(1 for dy, dx in c.rel
                   if 0 <= y0 + dy < h and 0 <= x0 + dx < w and int(self.grid[y0 + dy, x0 + dx]) == c.color)
        return hits * 5 >= len(c.rel) * 3

    def _mean(self, tie) -> tuple[float, float]:
        return (sum(self._ids[i][0] for i in tie) / len(tie), sum(self._ids[i][1] for i in tie) / len(tie))

    def _observe_bodies(self, frame: list[Comp]) -> None:
        """One frame with every handle placed: which sets of handles each thing
        that could be a body sits between."""
        objs = [o for o in self._objects(frame) if self._body_like(o)]
        n = len(self._ids)
        means: dict[frozenset, tuple[float, float]] = {}

        def mean(sub):
            m = means.get(sub)
            if m is None:
                m = means[sub] = self._mean(sub)
            return m

        # the bodies already tied: what sits where their handles have put them
        for tie, body in list(self._tied.items()):
            if any(i not in self._ids for i in tie):
                continue
            m = mean(tie)
            hit = self._seen_body(objs, body, m)
            if hit is not None:
                self._tied[tie] = replace(hit, tie=tie)  # its colours may have changed
            elif not _near(body, m) and any(o.sig == body.sig and o.place == body.place for o in objs):
                del self._tied[tie]  # it stayed put while its handles moved: not tied after all
                self.stats["untied"] += 1
        groups: dict[tuple, list[Obj]] = {}
        for o in objs:
            groups.setdefault(o.sig, []).append(o)
        subsets = None
        for sig, group in groups.items():
            self._sig_snaps[sig] += 1
            pts = [box_centre(o) for o in group]
            self._seen_at.setdefault(sig, set()).update(pts)

            def fits(sub):
                m = mean(sub)
                return any(abs(m[0] - p[0]) < 1 and abs(m[1] - p[1]) < 1 for p in pts)
            if sig not in self._fits_by:
                if subsets is None:
                    subsets = [frozenset(s) for size in range(1, min(MAX_TIE, n) + 1)
                               for s in combinations(range(n), size)]
                self._fits_by[sig] = {s for s in subsets if fits(s)}
            else:
                self._fits_by[sig] = {s for s in self._fits_by[sig] if fits(s)}
            self._resolve(sig, group, mean)

    def _resolve(self, sig: tuple, group: list[Obj], mean) -> None:
        """Tie the things of one look to handle sets, where the frames allow only one way."""
        fits = self._fits_by[sig]
        known = [t for t in fits if t in self._tied]
        free = [t for t in fits if t not in self._tied]
        need = len(group) - len(known)
        self._ambiguous.pop(sig, None)
        if need <= 0 or not free:
            return
        if self._snaps > 1 and len(self._seen_at[sig]) < 2:
            return  # new since the level began and never moved: a body half hidden, a piece cut by a line
        chosen = None
        if len(free) == need and _disjoint(free):
            chosen = free
        elif need == 1 and len(group) == 1 and len(self._seen_at[sig]) >= 2:
            # a body seen moving: of the sets that fit, the smallest, if only one
            smallest = min(len(t) for t in free)
            few = [t for t in free if len(t) == smallest]
            if len(few) == 1:
                chosen = few
        if chosen is None:
            self._ambiguous[sig] = sorted(free, key=len)
            return
        for t in chosen:
            o = next((o for o in group if _near(o, mean(t))), None)
            if o is not None:
                self._tied[t] = replace(o, tie=t)
                self.stats["ties"] += 1

    def _handles(self) -> dict[int, Comp]:
        return self._cur if self._handle_kind() is not None else {}

    def _pos(self, hs: dict[int, Comp], i: int) -> tuple[float, float]:
        """Where handle ``i`` is: its tracked centre (kept while it is hidden)."""
        return self._ids.get(i, box_centre(hs[i]))

    def _selected(self) -> int | None:
        """The handle drawn in the selected colour, read from the pixels in each
        handle's box (a handle partly under a body is still there)."""
        if not self.sel_colors or self.grid is None:
            return None
        col = self.sel_colors.most_common(1)[0][0]
        best = None
        for i, c in self._handles().items():
            cx, cy = self._ids.get(i, box_centre(c))
            x1, y1 = int(cx - (c.w - 1) / 2), int(cy - (c.h - 1) / 2)
            box = self.grid[max(y1, 0):max(y1 + c.h, 0), max(x1, 0):max(x1 + c.w, 0)]
            n = int((box == col).sum())
            if n and (best is None or n > best[0]):
                best = (n, i)
        return best[1] if best else None

    def _lines(self, comps: list[Comp] | None = None) -> set[int]:
        """Colours of the thin lines drawn between things and redrawn as they move:
        seen at a new place, nothing of theirs on screen more than a pixel wide (a
        diagonal line has a wide box but no more pixels than its length), and no
        tied body's (a body's colour may be left in thin pieces)."""
        comps = self.comps if comps is None else comps
        thick = {c.color for c in comps if c.area > max(c.w, c.h) + 1}
        for body in self._tied.values():
            thick |= body.colors
        return {col for col in self.moved_colors if col not in thick and col != self.bg}

    def _objects(self, comps: list[Comp], skip_goals: bool = True) -> list[Obj]:
        """The things in a frame: touching components of any colours, handles, the
        background, lines and (``skip_goals``) the goals' own dots left out."""
        cached = comps is self.comps and skip_goals
        if cached and self._obj_cache is not None and self._obj_cache[0] is comps:
            return self._obj_cache[1]
        kind = self._handle_kind()
        lines = self._lines(comps)
        goal_px = self._goal_pixels() if skip_goals else frozenset()
        # handles drawn into something they touch are cut out of it
        cut = {(h.x + dx, h.y + dy) for h in self._handles_in(comps) if h.kind == kind
               for dy, dx in h.rel} if kind is not None else set()
        owner: dict[tuple[int, int], int] = {}
        parts: list[tuple[Comp, list]] = []
        for c in comps:
            if c.kind == kind or c.color == self.bg or c.color in lines:
                continue
            px = [(c.x + dx, c.y + dy) for dy, dx in c.rel]
            if cut:
                px = [q for q in px if q not in cut]
                if not px:
                    continue
            if goal_px and all(p in goal_px for p in px):
                continue  # a goal's dots, or what is drawn over them
            for p in px:
                owner[p] = len(parts)
            parts.append((c, px))
        parent = list(range(len(parts)))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for (x, y), i in owner.items():
            for q in ((x + 1, y), (x, y + 1)):
                j = owner.get(q)
                if j is not None and j != i:
                    a, b = root(i), root(j)
                    if a != b:
                        parent[a] = b
        groups: dict[int, list[tuple[Comp, list]]] = {}
        for i, part in enumerate(parts):
            groups.setdefault(root(i), []).append(part)
        out = []
        for g in groups.values():
            px = [q for _, ps in g for q in ps]
            x1 = min(p[0] for p in px)
            y1 = min(p[1] for p in px)
            x2 = max(p[0] for p in px)
            y2 = max(p[1] for p in px)
            rel = frozenset((y - y1, x - x1) for x, y in px)
            out.append(Obj(x1, y1, x2 - x1 + 1, y2 - y1 + 1, rel, frozenset(c.color for c, _ in g)))
        if cached:
            self._obj_cache = (comps, out)
        return out

    @staticmethod
    def _body_like(o: Obj) -> bool:
        return o.area >= 4 and min(o.w, o.h) >= 2 and max(o.w, o.h) <= MAX_BODY

    def _first_objects(self) -> list[Obj]:
        key = (self._handle_kind(), frozenset(self._lines(self.first)))
        if self._first_cache is None or self._first_cache[0] != key:
            self._first_cache = (key, self._objects(self.first))
        return self._first_cache[1]

    def _goals(self) -> list[tuple[tuple[float, float], frozenset, frozenset]]:
        """Where bodies may have to go: clusters of small still things in the level's
        first frame, of any colours (handles, lines, and whatever could itself be a
        body left out), as (centre, colours, pixels)."""
        kind = self._handle_kind()
        if kind is None:
            return []
        lines = self._lines(self.first)
        key = (kind, frozenset(lines))
        if self._goal_cache is not None and self._goal_cache[0] == key:
            return self._goal_cache[1]
        boxes = [(c.x, c.y, c.x + c.w - 1, c.y + c.h - 1) for c in self.first if c.kind == kind]
        bulky: set = set()
        for o in self._objects(self.first, skip_goals=False):
            if self._body_like(o):
                bulky |= o.pixels
        small = []
        for c in self.first:
            if c.kind == kind or c.color == self.bg or c.color in lines or c.w > MARK_SIZE or c.h > MARK_SIZE:
                continue
            if any((c.x + dx, c.y + dy) in bulky for dy, dx in c.rel):
                continue  # part of something that could be a body
            if any(x1 <= c.x and c.x + c.w - 1 <= x2 and y1 <= c.y and c.y + c.h - 1 <= y2 for x1, y1, x2, y2 in boxes):
                continue  # a handle's own pixels
            small.append(c)
        out = []
        for cl in _clusters(small, GOAL_GAP):
            if len(cl) == 1 and cl[0].area <= 1:
                continue
            px = frozenset((c.x + dx, c.y + dy) for c in cl for dy, dx in c.rel)
            out.append((centre(cl), frozenset(c.color for c in cl), px))
        self._goal_cache = (key, out)
        return out

    def _goal_pixels(self) -> frozenset:
        goals = self._goals()
        return frozenset().union(*(g[2] for g in goals)) if goals else frozenset()

    @staticmethod
    def _best_goal(colors: frozenset, at: tuple[float, float], goals: list) -> tuple | None:
        """The goal for a thing of these colours as it is: all the goal's colours
        among its own, fewest of its own left over, most matched, nearest."""
        fit = [g for g in goals if g[1] <= colors]
        if not fit:
            return None
        return min(fit, key=lambda g: (len(colors - g[1]), -len(g[1]), _dist(at, g[0])))

    def _bodies(self) -> list[Obj]:
        """The tied bodies now: what sits where their handles put them, or, when it
        is hidden in part or drawn into something, its look as last seen whole,
        there."""
        if self._body_cache is not None and self._body_cache[0] is self.comps:
            return self._body_cache[1]
        objs = [o for o in self._objects(self.comps) if self._body_like(o)]
        out = []
        for tie, body in self._tied.items():
            if any(i not in self._ids for i in tie):
                continue
            m = self._mean(tie)
            hit = self._seen_body(objs, body, m)
            if hit is None:
                hit = replace(body, x=int(m[0] - (body.w - 1) / 2), y=int(m[1] - (body.h - 1) / 2))
            out.append(replace(hit, tie=tie))
        self._body_cache = (self.comps, out)
        return out

    @staticmethod
    def _seen_body(objs: list[Obj], body: Obj, m: tuple[float, float]) -> Obj | None:
        """The tied ``body`` as the frame shows it where its handles put it: whole, or
        drawn over in part (a line crossing it) but nowhere outside its outline; its
        colours are what shows. None when it is hidden or drawn into something."""
        hit = next((o for o in objs if o.rel == body.rel and _near(o, m)), None)
        if hit is not None:
            return hit
        top = (int(m[0] - (body.w - 1) / 2), int(m[1] - (body.h - 1) / 2))
        outline = {(top[0] + dx, top[1] + dy) for dy, dx in body.rel}
        for o in objs:
            if o.area * 5 >= body.area * 3 and o.pixels <= outline:
                return replace(body, x=top[0], y=top[1], colors=o.colors)
        return None

    def _pots(self) -> list[Obj]:
        """Still things a body might take colours from: things of a body's shape, as
        and where they were when the level began, that are no body."""
        first = {(o.sig, o.place) for o in self._first_objects() if self._body_like(o)}
        body_sigs = {b.sig for b in self._bodies()} | {b.sig for b in self._tied.values()}
        shapes = {b.rel for b in self._tied.values()}
        out = []
        for o in self._objects(self.comps):
            if o.rel not in shapes or (o.sig, o.place) not in first:
                continue
            if o.sig in body_sigs or o.sig in self._ambiguous or (o.sig, o.place) in self.bad_pots:
                continue
            out.append(o)
        return out

    def recognised(self) -> bool:
        return self._handle_kind() is not None and bool(self._tied)

    # -- planning ---------------------------------------------------------------

    def _allowed(self) -> set[int]:
        """Colours a handle or a body may be put over: the background and the thin
        lines drawn between handles and bodies (redrawn as they move); anything
        else (walls, hazards, other handles and bodies) is avoided."""
        return {self.bg} | self._lines()

    def _fits(self, c, top: tuple[int, int], own, allowed: set[int], forbid=frozenset()) -> bool:
        """Can ``c`` stand with its top-left at ``top``: on screen, over allowed
        colours only (its own pixels aside), on none of the ``forbid`` pixels?"""
        h, w = self.grid.shape
        for dy, dx in c.rel:
            x, y = top[0] + dx, top[1] + dy
            if not (0 <= x < w and 0 <= y < h):
                return False
            if (x, y) in own:
                continue
            if (x, y) in forbid:
                return False
            if self.mask is not None and self.mask[y, x]:
                return False
            if int(self.grid[y, x]) not in allowed:
                return False
        return True

    def _route(self, b: Obj, pots, goal_at: tuple[float, float]) -> tuple[float, tuple]:
        """The pots in the order a body visits them (nearest first) and the length of
        the way through them to the goal."""
        here, left, order, total = box_centre(b), list(pots), [], 0.0
        while left:
            p = min(left, key=lambda p: _dist(here, box_centre(p)))
            total += _dist(here, box_centre(p))
            here = box_centre(p)
            order.append(p)
            left.remove(p)
        return total + _dist(here, goal_at), tuple(order)

    def _assign(self, bodies: list[Obj]) -> dict[frozenset, tuple[int, tuple]]:
        """Each body's goal, and the pots it touches first: a goal whose colours the
        body has, or can take from pots that add only colours the goal has. The
        bodies go to different goals; as many as can, then fewest of their own
        colours left over, most matched, fewest pots, shortest way."""
        goals = self._goals()
        pots = self._pots() if self.absorb is not False else []
        opts: dict[frozenset, list] = {}
        for b in bodies:
            mine = []
            for gi, (gc, gcol, _) in enumerate(goals):
                extra = len(b.colors - gcol)
                need = gcol - b.colors
                if not need:
                    mine.append(((extra, -len(gcol), 0, _dist(box_centre(b), gc)), gi, ()))
                    continue
                usable = [p for p in pots if (p.colors - b.colors) and (p.colors - b.colors) <= gcol]
                for r in range(1, min(MAX_POTS, len(usable)) + 1):
                    found = [ps for ps in combinations(usable, r)
                             if need <= frozenset().union(*(p.colors for p in ps))]
                    if found:
                        length, order = min((self._route(b, ps, gc) for ps in found), key=lambda x: x[0])
                        mine.append(((extra, -len(gcol), r, length), gi, order))
                        break
            mine.sort(key=lambda o: o[0])
            opts[b.tie] = mine[:6]
        ties = [b.tie for b in bodies if opts[b.tie]]
        best: list = [None]

        def search(i, used_g, used_p, score, chosen):
            if i == len(ties):
                key = (-len(chosen), score)
                if best[0] is None or key < best[0][0]:
                    best[0] = (key, dict(chosen))
                return
            t = ties[i]
            for s, gi, ps in opts[t]:
                if gi in used_g or any(p.place in used_p for p in ps):
                    continue
                chosen[t] = (gi, ps)
                search(i + 1, used_g | {gi}, used_p | {p.place for p in ps},
                       tuple(a + b for a, b in zip(score, s)), chosen)
                del chosen[t]
            search(i + 1, used_g, used_p, score, chosen)

        search(0, frozenset(), frozenset(), (0, 0, 0, 0.0), {})
        return best[0][1] if best[0] else {}

    def _targets(self, f: Obj, goal) -> list[tuple[int, int]]:
        """Top-left places for body ``f`` that meet ``goal``, best first."""
        gc = goal[0]
        off = self.offsets.get((f.w, f.h))
        out = []
        if off is not None:
            cx, cy = gc[0] - off[0], gc[1] - off[1]
            out.append((0.0, (int(round(cx - (f.w - 1) / 2)), int(round(cy - (f.h - 1) / 2)))))
        for top in cover_places(f.rel, gc):
            c = (top[0] + (f.w - 1) / 2, top[1] + (f.h - 1) / 2)
            out.append((1.0 + abs(c[0] - gc[0]) + abs(c[1] - gc[1]), top))
        out.sort()
        seen, res = set(), []
        for _, top in out:
            if top not in seen and (f.tie, top[0], top[1]) not in self.bad_places:
                seen.add(top)
                res.append(top)
        return res

    def _plan(self) -> list[dict] | None:
        """The clicks of the cheapest known way to bring one body onto its goal (or
        onto a pot on its way there); [] when every body with a goal is on it."""
        hs = self._handles()
        if not hs or not self._tied:
            return None
        sel = self._selected()
        allowed = self._allowed()
        kind = self._handle_kind()
        boxes = {i: (c.x, c.y, c.x + c.w - 1, c.y + c.h - 1) for i, c in hs.items()}
        bodies = self._bodies()
        goals = self._goals()
        plan_of = self._assign(bodies)
        pots = self._pots()
        handle_px = frozenset((c.x + dx, c.y + dy) for c in hs.values() for dy, dx in c.rel)
        untied = [o for o in self._objects(self.comps) if self._body_like(o) and o.sig in self._ambiguous] if goals else []
        best = None
        all_done = not untied
        for f in bodies:
            if f.tie not in plan_of:
                continue
            gi, ps = plan_of[f.tie]
            pot = ps[0] if ps else None
            if pot is not None:
                pc = box_centre(pot)
                top = (int(round(pc[0] - (f.w - 1) / 2)), int(round(pc[1] - (f.h - 1) / 2)))
                targets = [top] if (f.tie, top[0], top[1]) not in self.bad_places else []
                at = None
                if not targets:
                    all_done = False
                    continue
            else:
                if self._on_goal(f, goals[gi]):
                    continue
                targets = self._targets(f, goals[gi])
                if not targets:
                    continue
                # a body already on a target moves only for a better one
                at = targets.index(f.place) if f.place in targets else None
                if at == 0:
                    continue
            better = targets[:at] if at is not None else targets
            tie = sorted(f.tie)
            if any(i not in hs for i in tie):
                if at is None:
                    all_done = False
                continue
            # where the body may stand: over the background, lines, its own colours and
            # handles (nothing stops a body passing over a handle), never on or against
            # another body or a pot other than the one it is going to
            forbid = set()
            for o in bodies:
                if o.tie != f.tie:
                    forbid |= _grown(o.pixels)
            for p in pots:
                if pot is None or p.place != pot.place:
                    forbid |= _grown(p.pixels)
            room = (f.pixels | handle_px | (pot.pixels if pot is not None else frozenset()),
                    allowed | f.colors, frozenset(forbid))
            info = {"pot": (pot.sig, pot.place) if pot is not None else None, "had": f.colors}
            k = len(tie)
            for top in better[:24]:
                if not self._fits(f, top, room[0], room[1], room[2]):
                    continue
                tc = (top[0] + (f.w - 1) / 2, top[1] + (f.h - 1) / 2)
                for h in tie:
                    p = self._place(hs, tie, h, tc, k, boxes, allowed, kind)
                    if p is None or p == self._pos(hs, h):
                        continue
                    cost = 1 + (h != sel)
                    if best is None or cost < best[0]:
                        best = (cost, [self._select(hs, h)] * (h != sel)
                                + [{"kind": "goto", "handle": h, "hkind": kind, "to": p,
                                    "body": (f.tie, top[0], top[1]), **info}])
                if best is not None and best[0] == 1:
                    break
            if best is None:
                two = self._two_moves(hs, tie, f, better, sel, boxes, allowed, kind, room)
                if two is not None:
                    best = (len(two), two)
            if best is None:
                every = self._gather(hs, tie, f, better, sel, boxes, allowed, kind, room)
                if every:
                    best = (len(every), every)
            if best is not None:
                if pot is not None and best[1][-1].get("body", (None, None, None))[1:] == top:
                    best[1][-1].update(info)
                break
            if at is None:
                all_done = False  # off every target and no way known onto one
        if best is not None:
            return best[1]
        if all_done and plan_of:
            return []
        if untied:
            return self._probe_tie(hs, untied[0], sel, boxes, allowed, kind)
        return None

    def _probe_tie(self, hs, f, sel, boxes, allowed, kind) -> list[dict] | None:
        """Several sets of handles fit thing ``f``: move a handle some of them hold
        and others do not, a little, and see whether the thing follows."""
        fits = self._ambiguous[f.sig]
        every = frozenset.intersection(*fits)
        some = sorted(frozenset.union(*fits) - every) or sorted(frozenset.union(*fits))
        for h in some:
            if h not in hs:
                continue
            c = self._pos(hs, h)
            near = sorted(((int(c[0]) + dx, int(c[1]) + dy) for dx in range(-8, 9) for dy in range(-8, 9)
                           if abs(dx) + abs(dy) >= 3), key=lambda q: (abs(q[0] - c[0]) + abs(q[1] - c[1]), q))
            for q in near:
                if self._handle_can_go(hs[h], q, boxes, allowed, kind, h):
                    self.stats["tie_probes"] += 1
                    return [self._select(hs, h)] * (h != sel) + [
                        {"kind": "goto", "handle": h, "hkind": kind, "to": q, "body": None}]
        return None

    def _place(self, hs, tie, h, tc, k, boxes, allowed, kind, moved=None) -> tuple[int, int] | None:
        """Where handle ``h`` must go for the tied body's centre to be ``tc``, if it can go there."""
        pos = {i: self._pos(hs, i) for i in tie}
        if moved:
            pos.update(moved)
        px = k * tc[0] - sum(pos[i][0] for i in tie if i != h)
        py = k * tc[1] - sum(pos[i][1] for i in tie if i != h)
        if abs(px - round(px)) > 1e-6 or abs(py - round(py)) > 1e-6:
            return None
        p = (int(round(px)), int(round(py)))
        if not self._handle_can_go(hs[h], p, boxes, allowed, kind, h):
            return None
        return p

    def _handle_can_go(self, c: Comp, p, boxes, allowed, kind, h) -> bool:
        """Can handle ``c`` be sent to ``p``: not where it failed to go, not onto
        another handle (a click there selects it), over the background and the
        lines only (a handle on a goal or a body is hidden in part)?"""
        if (kind, p[0], p[1]) in self.bad_points:
            return False
        for i, (x1, y1, x2, y2) in boxes.items():
            if i != h and x1 - 1 <= p[0] <= x2 + 1 and y1 - 1 <= p[1] <= y2 + 1:
                return False  # a click there selects that handle instead
        top = (int(round(p[0] - (c.w - 1) / 2)), int(round(p[1] - (c.h - 1) / 2)))
        own = {(c.x + dx, c.y + dy) for dy, dx in c.rel}
        if not self._fits(c, top, own, allowed):
            return False
        # not touching a body or a pot where it has the colour of a handle on the
        # move (the selected one): the two would be drawn as one
        solid = self._solid()
        moving = self.sel_colors.most_common(1)[0][0] if self.sel_colors else c.color
        there = {(top[0] + dx, top[1] + dy) for dy, dx in c.rel}
        for x, y in there:
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if q in there or q in own or q not in solid:
                    continue
                if int(self.grid[q[1], q[0]]) == moving:
                    return False
        return True

    def _solid(self) -> frozenset:
        """The pixels of everything on screen that could be a body (bodies and pots)."""
        if self._solid_cache is None or self._solid_cache[0] is not self.comps:
            px = frozenset().union(*(o.pixels for o in self._objects(self.comps) if self._body_like(o)))
            self._solid_cache = (self.comps, px)
        return self._solid_cache[1]

    def _two_moves(self, hs, tie, f, targets, sel, boxes, allowed, kind, room) -> list[dict] | None:
        """Move one tied handle somewhere free, then another onto the solution."""
        k = len(tie)
        if k < 2:
            return None
        own_f, body_ok, forbid = room
        h_grid, w_grid = self.grid.shape
        for top in targets[:6]:
            if not self._fits(f, top, own_f, body_ok, forbid):
                continue
            tc = (top[0] + (f.w - 1) / 2, top[1] + (f.h - 1) / 2)
            for h1 in tie:
                pts = sorted(((x, y) for x in range(2, w_grid - 2, LATTICE) for y in range(2, h_grid - 2, LATTICE)),
                             key=lambda q: abs(q[0] - tc[0]) + abs(q[1] - tc[1]))
                tried = 0
                for q in pts:
                    if tried > 400:
                        break
                    if not self._handle_can_go(hs[h1], q, boxes, allowed, kind, h1):
                        continue
                    tried += 1
                    # the body after the first move must not land on something still
                    pos = {i: self._pos(hs, i) for i in tie}
                    pos[h1] = q
                    mid = (sum(p[0] for p in pos.values()) / k, sum(p[1] for p in pos.values()) / k)
                    mtop = (int(mid[0] - (f.w - 1) / 2), int(mid[1] - (f.h - 1) / 2))
                    if (f.tie, mtop[0], mtop[1]) in self.bad_places or not self._fits(f, mtop, own_f, body_ok, forbid):
                        continue
                    c1 = hs[h1]
                    moved_boxes = dict(boxes)
                    moved_boxes[h1] = (q[0] - (c1.w - 1) // 2, q[1] - (c1.h - 1) // 2,
                                       q[0] + c1.w // 2, q[1] + c1.h // 2)
                    for h2 in tie:
                        if h2 == h1:
                            continue
                        p = self._place(hs, tie, h2, tc, k, moved_boxes, allowed, kind, moved={h1: q})
                        if p is None:
                            continue
                        steps = []
                        if h1 != sel:
                            steps.append(self._select(hs, h1))
                        steps.append({"kind": "goto", "handle": h1, "hkind": kind, "to": q,
                                      "body": (f.tie, mtop[0], mtop[1])})
                        return steps  # the second move is planned again after the first
        return None

    def _gather(self, hs, tie, f, targets, sel, boxes, allowed, kind, room) -> list[dict] | None:
        """Every tied handle to a place round the target whose mean is the target,
        in an order that never puts the body on something still on the way."""
        k = len(tie)
        own_f, body_ok, forbid = room
        others = {i: b for i, b in boxes.items() if i not in tie}
        dirs = [(cos(2 * pi * a / 8), sin(2 * pi * a / 8)) for a in range(8)]
        for top in targets[:6]:
            if not self._fits(f, top, own_f, body_ok, forbid):
                continue
            tc = (top[0] + (f.w - 1) / 2, top[1] + (f.h - 1) / 2)
            # first the handles spread evenly round the target
            for d in (6, 7, 8, 10, 12, 14):
                for turn in (0.0, 0.25, 0.5, 0.75):
                    final = {}
                    for j, h in enumerate(tie[:-1]):
                        a = 2 * pi * (j / k + turn / k)
                        final[h] = (int(round(tc[0] + d * cos(a))), int(round(tc[1] + d * sin(a))))
                    lx = k * tc[0] - sum(p[0] for p in final.values())
                    ly = k * tc[1] - sum(p[1] for p in final.values())
                    if abs(lx - round(lx)) > 1e-6 or abs(ly - round(ly)) > 1e-6:
                        continue
                    final[tie[-1]] = (int(round(lx)), int(round(ly)))
                    if not self._apart(final, hs):
                        continue
                    plan = self._order(hs, tie, f, final, sel, others, allowed, kind, body_ok, forbid, own_f)
                    if plan is not None:
                        return plan
            # then any places on rings round the target, nearest to each handle first
            cands = {}
            for h in tie[:-1]:
                here = self._pos(hs, h)
                pts = {(int(round(tc[0] + d * dx)), int(round(tc[1] + d * dy)))
                       for d in (6, 8, 10, 12, 14, 16) for dx, dy in dirs}
                ok = [q for q in pts if self._handle_can_go(hs[h], q, others, allowed, kind, h)]
                ok.sort(key=lambda q: abs(q[0] - here[0]) + abs(q[1] - here[1]))
                cands[h] = ok[:10]
            tried = 0
            for combo in self._combos([cands[h] for h in tie[:-1]]):
                tried += 1
                if tried > 3000:
                    break
                final = dict(zip(tie[:-1], combo))
                lx = k * tc[0] - sum(p[0] for p in combo)
                ly = k * tc[1] - sum(p[1] for p in combo)
                if abs(lx - round(lx)) > 1e-6 or abs(ly - round(ly)) > 1e-6:
                    continue
                final[tie[-1]] = (int(round(lx)), int(round(ly)))
                if not self._apart(final, hs):
                    continue
                plan = self._order(hs, tie, f, final, sel, others, allowed, kind, body_ok, forbid, own_f)
                if plan is not None:
                    return plan
        return None

    @staticmethod
    def _combos(lists: list[list]):
        """Combinations of one item from each list, items early in their lists first."""
        if not lists:
            yield ()
            return
        for depth in range(max((len(x) for x in lists), default=0)):
            yield from _product_upto(lists, depth)

    @staticmethod
    def _apart(final: dict, hs: dict) -> bool:
        """Final handle places far enough apart that no two boxes touch."""
        items = list(final.items())
        for i, (a, p) in enumerate(items):
            for b, q in items[i + 1:]:
                if abs(p[0] - q[0]) <= (hs[a].w + hs[b].w) // 2 + 1 and abs(p[1] - q[1]) <= (hs[a].h + hs[b].h) // 2 + 1:
                    return False
        return True

    def _order(self, hs, tie, f, final, sel, others, allowed, kind, body_ok, forbid, own_f) -> list[dict] | None:
        k = len(tie)
        for order in permutations(tie):
            pos = {i: self._pos(hs, i) for i in tie}
            steps, ok, cur_sel = [], True, sel
            for h in order:
                c = hs[h]
                if self._pos(hs, h) == final[h]:
                    continue  # already there
                bx = dict(others)
                for i in tie:
                    if i != h:
                        q = pos[i]
                        bx[i] = (int(q[0]) - (hs[i].w - 1) // 2, int(q[1]) - (hs[i].h - 1) // 2,
                                 int(q[0]) + hs[i].w // 2, int(q[1]) + hs[i].h // 2)
                if not self._handle_can_go(c, final[h], bx, allowed, kind, h):
                    ok = False
                    break
                pos[h] = final[h]
                mid = (sum(p[0] for p in pos.values()) / k, sum(p[1] for p in pos.values()) / k)
                mtop = (int(mid[0] - (f.w - 1) / 2), int(mid[1] - (f.h - 1) / 2))
                if (f.tie, mtop[0], mtop[1]) in self.bad_places or not self._fits(f, mtop, own_f, body_ok, forbid):
                    ok = False
                    break
                if h != cur_sel:
                    steps.append(self._select(hs, h))
                    cur_sel = h
                steps.append({"kind": "goto", "handle": h, "hkind": kind, "to": final[h],
                              "body": (f.tie, mtop[0], mtop[1])})
            if ok:
                return steps
        return None

    def _select(self, hs, h) -> dict:
        """A click on a pixel the handle shows (not one that failed to select it)."""
        c = hs[h]
        pts = sorted(((c.x + dx, c.y + dy) for dy, dx in c.rel),
                     key=lambda q: (abs(q[0] - c.point[0]) + abs(q[1] - c.point[1]), q))
        at = next((q for q in pts if ("select", q[0], q[1]) not in self.bad_points), c.point)
        return {"kind": "select", "handle": h, "at": (int(at[0]), int(at[1]))}

    def _on_goal(self, f: Obj, goal) -> bool:
        """Is body ``f`` on ``goal``: where the learned relation puts it, or over the
        goal while the goal looks different from when the level began (the game's
        sign that it is met)?"""
        off = self.offsets.get((f.w, f.h))
        fc, gc = box_centre(f), goal[0]
        if off is not None and abs(gc[0] - off[0] - fc[0]) < 0.6 and abs(gc[1] - off[1] - fc[1]) < 0.6:
            return True
        body = f.pixels
        if not body & goal[2] or self.first_grid is None:
            return False
        return any(int(self.grid[y, x]) != int(self.first_grid[y, x]) for x, y in goal[2] - body)

    # -- acting -----------------------------------------------------------------

    def _give_up(self, why: str) -> None:
        self.stats["give_up_" + why] += 1
        self.gave_up = True
        self.in_control = False
        self.strikes += 1
        if self.strikes >= (3 if self.confirmed else 2):
            self.disabled = True

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if 6 not in obs.available_actions or not self.recognised():
            return None
        plan = self._plan()
        if plan == []:
            # every body stands where the solver thinks it should and the level goes
            # on: those places are not the goal
            for f in self._bodies():
                self.bad_places.add((f.tie, f.x, f.y))
            plan = self._plan()
        if not plan:
            if self.in_control:
                self._give_up("no_plan")
            return None
        if not self.in_control:
            self.in_control = True
            self.stats["takeovers"] += 1
        limit = MAX_LEVEL_ACTIONS if self.confirmed else UNPROVEN_ACTIONS
        if self.level_actions >= limit:
            self._give_up("actions")
            return None
        if self.fails >= MAX_FAILS:
            self._give_up("fails")
            return None
        step = plan[0]
        self.level_actions += 1
        self.stats["clicks"] += 1
        self.expect = step
        x, y = step["at"] if step["kind"] == "select" else step["to"]
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}
