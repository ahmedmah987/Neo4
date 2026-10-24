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
* a *body* is a component of its own colour and shape that moved with the
  handles and whose centre has been, in every frame seen, the mean of the
  centres of the same set of handles;
* a body's goal is the cluster of small still marks of its colour (handles'
  own pixels left out); when a level ends, where the body stood relative to its
  mark is kept for the next levels, otherwise any place where the body covers
  the mark's centre is tried, nearest first.

To put a body on a target with ``k`` handles it moves one handle to ``k`` times
the target minus the other handles' centres, selecting it first if needed; if no
handle can go there (off screen, onto something still, onto another handle), it
moves one handle somewhere free first and then another. Places a handle could
not go and places the body bounced from are not tried again.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
from math import cos, pi, sin

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from . import frame as fp
from .marker import Comp, components, cover_places, mark_clusters, centre

Action = tuple[GameAction, dict[str, int] | None]

GOTO_PROOF = 2  # clicks that sent a handle to the clicked point before the rule is trusted
MAX_HANDLES = 8  # handles a body may be tied to are searched among this many
MAX_TIE = 4  # a body is tied to at most this many handles
MAX_LEVEL_ACTIONS = 120  # actions the solver may spend on one level
UNPROVEN_ACTIONS = 40  # ... in a game it has not completed a level of
MAX_FAILS = 8  # clicks that did not do what was planned, per level
LATTICE = 2  # spacing of the places tried for a handle moved out of the way
MARK_SIZE = 12


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


def box_centre(c: Comp) -> tuple[float, float]:
    return (c.x + (c.w - 1) / 2.0, c.y + (c.h - 1) / 2.0)


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
        # where a body stood relative to its mark when a level ended: (w, h) -> centre offset
        self.offsets: dict[tuple, tuple[float, float]] = {}
        self.level = -1
        self.in_control = False
        self._new_level(None)

    # -- per level -------------------------------------------------------------

    def _new_level(self, grid: np.ndarray | None) -> None:
        self.grid = grid
        self.mask = fp.status_bar_mask(fp.segment(grid)) if grid is not None else None
        self.bg = int(np.bincount(grid.ravel()).argmax()) if grid is not None else -1
        self.comps: list[Comp] = components(grid, self.mask) if grid is not None else []
        self.first = list(self.comps)
        # frames kept until the handle kind is known, to number the handles from the start
        self.history: list[list[Comp]] = [self.comps] if grid is not None else []
        self._track_kind: str | None = None
        self._ids: dict[int, tuple[float, float]] = {}  # handle number -> centre, now
        self._cur: dict[int, Comp] = {}  # handle number -> component, now
        self._body_area: dict[int, int] = {}  # body colour -> its area seen whole
        self._body_shape: dict[int, tuple] = {}  # body colour -> (pixels, w, h) seen whole
        self._snaps = 0  # frames seen with every handle placed
        self._fits_by: dict[int, set] = {}  # body colour -> handle sets that fit every such frame
        self._seen_at: dict[int, set] = {}  # body colour -> places it was seen at
        self.moving_colors: set[int] = set()
        self.bad_points: set = set()  # (kind, x, y): a handle sent there did not go
        self.bad_places: set = set()  # (colour, x, y): a body placed there bounced or did not end the level
        self.level_actions = 0
        self.fails = 0
        self.gave_up = False
        self.expect: dict | None = None
        self._fit_cache = None
        self._ambiguous: dict[tuple, list[frozenset]] = {}
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
        grid = np.asarray(obs.frame[-1])
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0:
                if len(obs.frame) > 1:
                    self._learn_offsets(np.asarray(obs.frame[-2]))
                if self.in_control:
                    self.confirmed = True
                    self.stats["levels"] += 1
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level(grid)
            return
        prev = self.comps
        after = components(grid, self.mask)
        self.grid = grid
        self.comps = after
        self.comps_key = frozenset((c.color, c.shape, c.x, c.y) for c in after)  # the status bar left out
        self._fit_cache = None
        if action is None or action is GameAction.RESET:
            # a fresh attempt: the handles are back where the level began; what the
            # frames taught about the ties stays
            if self._track_kind is not None:
                start = [c for c in self.first if c.kind == self._track_kind]
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
                self.moving_colors.add(c.color)
        if data is not None and action.is_complex():
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
        self._sync(after)
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
        if body is not None and any((f.color, f.x, f.y) == body for f in self._bodies(self.comps)):
            return  # the body is where it was sent
        if h is None or box_centre(h) != exp["to"]:
            # the handle did not go there (a wall), or went and came back (the body bounced)
            self.bad_points.add((exp["hkind"], exp["to"][0], exp["to"][1]))
            if exp.get("body") is not None:
                self.bad_places.add(exp["body"])
            self.fails += 1
            self.stats["mismatches"] += 1

    def _learn_offsets(self, final: np.ndarray) -> None:
        """Where each body stood relative to the nearest mark of its colour when
        the level ended."""
        comps = components(final, self.mask)
        ties = self._ties()
        for f in self._bodies(comps):
            if f.color not in ties:
                continue
            marks = self._marks(f.color)
            if not marks:
                continue
            fc = box_centre(f)
            m = min(marks, key=lambda p: abs(p[0] - fc[0]) + abs(p[1] - fc[1]))
            if abs(m[0] - fc[0]) + abs(m[1] - fc[1]) <= f.w + f.h:
                self.offsets[(f.w, f.h)] = (m[0] - fc[0], m[1] - fc[1])
                self.stats["offsets"] += 1

    # -- the model --------------------------------------------------------------

    def _handle_kind(self) -> str | None:
        if not self.goto_kinds:
            return None
        kind, n = self.goto_kinds.most_common(1)[0]
        return kind if n >= GOTO_PROOF else None

    def _sync(self, latest: list[Comp]) -> None:
        """Number the handles: from the level's first frame on, a handle keeps its
        number while it stays where it was, the one that moved takes the number left
        over. Built from the kept frames once the handle kind is known, then frame
        by frame."""
        kind = self._handle_kind()
        if kind is None:
            return
        if kind != self._track_kind:
            self._track_kind = kind
            self._ids, self._cur, self._snaps = {}, {}, 0
            self._fits_by, self._seen_at = {}, {}
            start = [c for c in self.first if c.kind == kind]
            if not start or len(start) > MAX_HANDLES:
                return
            self._ids = {i: box_centre(c) for i, c in enumerate(start)}
            self._cur = dict(enumerate(start))
            for frame in self.history:
                self._advance(frame)
            self.history = []
            return
        self._advance(latest)

    def _advance(self, frame: list[Comp]) -> None:
        if not self._ids:
            return
        hs = [c for c in frame if c.kind == self._track_kind]
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
        for i in left:
            if i in self._cur:
                cur[i] = self._cur[i]
        self._cur = cur
        self._ids = {i: box_centre(c) for i, c in cur.items()}
        if was_sel is None or was_sel not in left:
            # a hidden handle that was not selected has not moved: the frame says where
            # every handle is unless the selected one is hidden
            self._snaps += 1
            n = len(self._ids)
            for c in self._bodies(frame):
                full = self._body_area.get(c.color, 0)
                if c.area < full:
                    continue  # partly hidden: its box centre is not its centre
                self._body_area[c.color] = c.area
                self._body_shape[c.color] = (c.rel, c.w, c.h)
                p = box_centre(c)
                self._seen_at.setdefault(c.color, set()).add(p)

                def fits(sub):
                    return (abs(sum(self._ids[i][0] for i in sub) / len(sub) - p[0]) < 1
                            and abs(sum(self._ids[i][1] for i in sub) / len(sub) - p[1]) < 1)
                if c.color not in self._fits_by:
                    self._fits_by[c.color] = {frozenset(sub) for size in range(1, min(MAX_TIE, n) + 1)
                                              for sub in combinations(range(n), size) if fits(sub)}
                else:
                    self._fits_by[c.color] = {sub for sub in self._fits_by[c.color] if fits(sub)}
        self._fit_cache = None

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

    def _bodies(self, comps: list[Comp]) -> list[Comp]:
        """For each colour, its largest blob (not a handle, not a line): what may be
        a body tied to handles. A body is known by its colour, so a body partly
        hidden under a handle or a line is still the same body."""
        kind = self._handle_kind()
        best: dict[int, Comp] = {}
        for c in comps:
            if c.kind == kind or c.area < 4 or min(c.w, c.h) < 2 or c.color == self.bg:
                continue
            if c.color not in best or c.area > best[c.color].area:
                best[c.color] = c
        if comps is self.comps and self._body_shape:
            # a tied body grown into something of its colour (its ring) is where its
            # handles put it
            for col, tie in self._ties().items():
                c, whole = best.get(col), self._body_shape.get(col)
                if c is None or whole is None or c.area <= self._body_area.get(col, 0):
                    continue
                if any(i not in self._ids for i in tie):
                    continue
                rel, w, h = whole
                cx = sum(self._ids[i][0] for i in tie) / len(tie)
                cy = sum(self._ids[i][1] for i in tie) / len(tie)
                x, y = int(cx - (w - 1) / 2), int(cy - (h - 1) / 2)
                best[col] = Comp(col, c.shape, c.kind, x, y, w, h, len(rel), rel, (x, y), c.id)
        return list(best.values())

    def _ties(self) -> dict[int, frozenset]:
        """For each body colour, the set of handles whose mean centre has been the
        body's centre in every frame since the handles were known."""
        if self._fit_cache is not None:
            return self._fit_cache
        out: dict[int, frozenset] = {}
        self._ambiguous = {}
        for col, fits in self._fits_by.items():
            if not fits:
                continue
            if len(fits) == 1:
                out[col] = next(iter(fits))
            elif len(self._seen_at.get(col, ())) >= 2:
                # a body seen moving: of the sets that fit, the smallest, if only one
                smallest = min(len(f) for f in fits)
                few = [f for f in fits if len(f) == smallest]
                if len(few) == 1:
                    out[col] = few[0]
                else:
                    self._ambiguous[col] = few
            else:
                self._ambiguous[col] = sorted(fits, key=len)
        self._fit_cache = out
        return out

    def _marks(self, color: int) -> list[tuple[float, float]]:
        """Centres of the clusters of small still components of ``color`` in the
        level's first frame, handles and what lies inside them left out."""
        kind = self._handle_kind()
        boxes = [(c.x, c.y, c.x + c.w - 1, c.y + c.h - 1) for c in self.first if c.kind == kind]
        still = []
        bodies = set(self._ties())
        for c in self.first:
            if c.color != color or c.w > MARK_SIZE or c.h > MARK_SIZE or c.kind == kind:
                continue
            if c.color in bodies and c.area >= 4 and min(c.w, c.h) >= 2:
                continue  # a body where the level began is not its own goal
            if any(x1 <= c.x and c.x + c.w - 1 <= x2 and y1 <= c.y and c.y + c.h - 1 <= y2 for x1, y1, x2, y2 in boxes):
                continue  # a handle's own pixels
            still.append(c)
        return [centre(cl) for cl in mark_clusters(still) if len(cl) > 1 or cl[0].area > 1]

    def _mark_colors(self) -> set[int]:
        return {col for col in self._ties() if self._marks(col)}

    def recognised(self) -> bool:
        return self._handle_kind() is not None and bool(self._ties())

    # -- planning ---------------------------------------------------------------

    def _allowed(self) -> set[int]:
        """Colours a handle or a body may be put over: the background and the thin
        lines drawn between handles and bodies (redrawn as they move); anything
        else (walls, hazards, other handles and bodies) is avoided."""
        out = {self.bg}
        for col in self.moving_colors:
            if all(min(c.w, c.h) <= 2 for c in self.comps if c.color == col):
                out.add(col)
        return out

    def _fits(self, c: Comp, top: tuple[int, int], own: set, allowed: set[int]) -> bool:
        """Can ``c`` stand with its top-left at ``top``: on screen, over allowed
        colours only (its own pixels aside)?"""
        h, w = self.grid.shape
        for dy, dx in c.rel:
            x, y = top[0] + dx, top[1] + dy
            if not (0 <= x < w and 0 <= y < h):
                return False
            if (x, y) in own:
                continue
            if self.mask is not None and self.mask[y, x]:
                return False
            if int(self.grid[y, x]) not in allowed:
                return False
        return True

    def _targets(self, f: Comp) -> list[tuple[int, int]]:
        """Top-left places for body ``f`` that meet a mark of its colour, best first."""
        marks = self._marks(f.color)
        off = self.offsets.get((f.w, f.h))
        out = []
        for m in marks:
            if off is not None:
                cx, cy = m[0] - off[0], m[1] - off[1]
                top = (int(round(cx - (f.w - 1) / 2)), int(round(cy - (f.h - 1) / 2)))
                out.append((0.0, top))
            for top in cover_places(f.rel, m):
                c = (top[0] + (f.w - 1) / 2, top[1] + (f.h - 1) / 2)
                out.append((1.0 + abs(c[0] - m[0]) + abs(c[1] - m[1]), top))
        out.sort()
        seen, res = set(), []
        for _, top in out:
            if top not in seen and (f.color, top[0], top[1]) not in self.bad_places:
                seen.add(top)
                res.append(top)
        return res

    def _plan(self) -> list[dict] | None:
        """The clicks of the cheapest known way to bring one body onto a mark; []
        when every body with a mark is on one."""
        hs = self._handles()
        ties = self._ties()
        if not hs or not ties:
            return None
        sel = self._selected()
        allowed = self._allowed()
        kind = self._handle_kind()
        boxes = {i: (c.x, c.y, c.x + c.w - 1, c.y + c.h - 1) for i, c in hs.items()}
        bodies = [f for f in self._bodies(self.comps) if f.color in ties]
        untied = [f for f in self._bodies(self.comps) if f.color in self._ambiguous
                  and f.color not in ties and self._marks(f.color)]
        best = None
        all_done = not untied
        for f in bodies:
            targets = self._targets(f)
            if not targets or self._on_mark(f):
                continue
            # a body already on a target moves only for a better one
            at = targets.index((f.x, f.y)) if (f.x, f.y) in targets else None
            if at == 0:
                continue
            better = targets[:at] if at is not None else targets
            tie = sorted(ties[f.color])
            if any(i not in hs for i in tie):
                if at is None:
                    all_done = False
                continue
            k = len(tie)
            own_f = {(f.x + dx, f.y + dy) for dy, dx in f.rel}
            for top in better[:24]:
                if not self._fits(f, top, own_f, allowed | {f.color}):
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
                                    "body": (f.color, top[0], top[1])}])
                if best is not None and best[0] == 1:
                    break
            if best is None:
                two = self._two_moves(hs, tie, f, better, sel, boxes, allowed, kind)
                if two is not None:
                    best = (len(two), two)
            if best is None:
                every = self._gather(hs, tie, f, better, sel, boxes, allowed, kind)
                if every:
                    best = (len(every), every)
            if best is not None:
                break
            if at is None:
                all_done = False  # off every target and no way known onto one
        if best is not None:
            return best[1]
        if all_done and bodies:
            return []
        if untied:
            return self._probe_tie(hs, untied[0], sel, boxes, allowed, kind)
        return None

    def _probe_tie(self, hs, f, sel, boxes, allowed, kind) -> list[dict] | None:
        """Several sets of handles fit body ``f``: move a handle some of them hold
        and others do not, a little, and see whether the body follows."""
        fits = self._ambiguous[f.color]
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
        lines only (a handle on a mark or a body is hidden in part)?"""
        if (kind, p[0], p[1]) in self.bad_points:
            return False
        for i, (x1, y1, x2, y2) in boxes.items():
            if i != h and x1 - 1 <= p[0] <= x2 + 1 and y1 - 1 <= p[1] <= y2 + 1:
                return False  # a click there selects that handle instead
        top = (int(round(p[0] - (c.w - 1) / 2)), int(round(p[1] - (c.h - 1) / 2)))
        own = {(c.x + dx, c.y + dy) for dy, dx in c.rel}
        return self._fits(c, top, own, allowed)

    def _two_moves(self, hs, tie, f, targets, sel, boxes, allowed, kind) -> list[dict] | None:
        """Move one tied handle somewhere free, then another onto the solution."""
        k = len(tie)
        if k < 2:
            return None
        own_f = {(f.x + dx, f.y + dy) for dy, dx in f.rel}
        h_grid, w_grid = self.grid.shape
        for top in targets[:6]:
            if not self._fits(f, top, own_f, allowed | {f.color}):
                continue
            tc = (top[0] + (f.w - 1) / 2, top[1] + (f.h - 1) / 2)
            for h1 in tie:
                c1 = self._pos(hs, h1)
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
                    if (f.color, mtop[0], mtop[1]) in self.bad_places or not self._fits(f, mtop, own_f, allowed | {f.color}):
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
                                      "body": (f.color, mtop[0], mtop[1])})
                        return steps  # the second move is planned again after the first
        return None

    def _gather(self, hs, tie, f, targets, sel, boxes, allowed, kind) -> list[dict] | None:
        """Every tied handle to a place round the target whose mean is the target,
        in an order that never puts the body on something still on the way."""
        k = len(tie)
        own_f = {(f.x + dx, f.y + dy) for dy, dx in f.rel}
        body_ok = allowed | {f.color} | self._mark_colors()
        others = {i: b for i, b in boxes.items() if i not in tie}
        dirs = [(cos(2 * pi * a / 8), sin(2 * pi * a / 8)) for a in range(8)]
        for top in targets[:6]:
            if not self._fits(f, top, own_f, body_ok):
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
                    plan = self._order(hs, tie, f, final, sel, others, allowed, kind, body_ok, own_f)
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
                plan = self._order(hs, tie, f, final, sel, others, allowed, kind, body_ok, own_f)
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

    def _order(self, hs, tie, f, final, sel, others, allowed, kind, body_ok, own_f) -> list[dict] | None:
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
                if (f.color, mtop[0], mtop[1]) in self.bad_places or not self._fits(f, mtop, own_f, body_ok):
                    ok = False
                    break
                if h != cur_sel:
                    steps.append(self._select(hs, h))
                    cur_sel = h
                steps.append({"kind": "goto", "handle": h, "hkind": kind, "to": final[h],
                              "body": (f.color, mtop[0], mtop[1])})
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

    def _on_mark(self, f: Comp) -> bool:
        """Is body ``f`` where the learned relation, or failing that a cover, puts it?"""
        off = self.offsets.get((f.w, f.h))
        fc = box_centre(f)
        for m in self._marks(f.color):
            if off is not None:
                if abs(m[0] - off[0] - fc[0]) < 0.6 and abs(m[1] - off[1] - fc[1]) < 0.6:
                    return True
        return False

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
            for f in self._bodies(self.comps):
                self.bad_places.add((f.color, f.x, f.y))
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
