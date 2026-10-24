"""The translation family: rewrite a row of glyphs by the rules shown.

Some puzzles show rules, each a sequence of glyphs with an arrow to another
sequence, an input row, and an output row the player edits: a cursor stands on
one glyph of the output, two keys move it, two others turn the glyph into the
next or previous one of its family. The output must be the input rewritten rule by
rule. Every glyph may be drawn turned.

From the frames alone:

* a glyph is a square tile of one frame colour with a pattern of another colour
  inside; it is known by its frame colour (its family) and its pattern up to a
  quarter turn;
* tiles side by side form sequences; two sequences on one row joined by a line
  across the gap between them are a rule, the left one rewritten as the right one;
* the other sequences are the rows; the editable one carries the cursor (marks in a
  colour no tile has, just above and below one of its tiles), the input is the
  other;
* the target is the input read left to right, at each place the first rule (in
  reading order) whose left side is there giving its right side; a right side in a
  family that is not the output's is rewritten again, by the rule whose left side it
  is (or glyph by glyph);
* each family's cycle (what a key turns a glyph into) is learned by watching it.

The solver walks the cursor to the nearest wrong glyph and turns it towards its
target, the shorter way round when the cycle is known, forward otherwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

Action = tuple[GameAction, dict[str, int] | None]

TILE = 7  # a glyph tile is this many pixels square, tiles of a sequence this far apart
MAX_FAILS = 6  # actions that did not do what was expected, per level
MAX_LEVEL_ACTIONS = 200


@dataclass(frozen=True)
class Glyph:
    family: int  # the tile's frame colour
    shape: bytes  # the pattern, the same however the tile is turned
    x: int
    y: int

    @property
    def id(self) -> tuple[int, bytes]:
        return (self.family, self.shape)


def _canon(mask: np.ndarray) -> bytes:
    return min(np.packbits(np.rot90(mask, k)).tobytes() for k in range(4))


def find_glyphs(grid: np.ndarray) -> list[Glyph]:
    """Square tiles: a ring of one colour round a pattern of exactly one other."""
    h, w = grid.shape
    out = []
    taken = np.zeros((h, w), dtype=bool)
    n = TILE
    for y in range(h - n + 1):
        for x in range(w - n + 1):
            if taken[y, x]:
                continue
            f = grid[y, x]
            if grid[y, x + n - 1] != f or grid[y + n - 1, x] != f or grid[y + n - 1, x + n - 1] != f:
                continue
            if (y > 0 and grid[y - 1, x + n // 2] == f) or (y + n < h and grid[y + n, x + n // 2] == f):
                continue  # a ring of what surrounds it is no tile
            tile = grid[y:y + n, x:x + n]
            ring = np.ones((n, n), dtype=bool)
            ring[1:-1, 1:-1] = False
            if not (tile[ring] == f).all():
                continue
            inner = tile[1:-1, 1:-1]
            others = set(np.unique(inner).tolist()) - {int(f)}
            if len(others) != 1:
                continue
            g = others.pop()
            out.append(Glyph(int(f), _canon(inner == g), x, y))
            taken[y:y + n, x:x + n] = True
    return out


def sequences(glyphs: list[Glyph]) -> list[list[Glyph]]:
    """Tiles on one row, each the next one's neighbour, left to right."""
    rows: dict[int, list[Glyph]] = {}
    for g in glyphs:
        rows.setdefault(g.y, []).append(g)
    out = []
    for y in sorted(rows):
        row = sorted(rows[y], key=lambda g: g.x)
        cur = [row[0]]
        for g in row[1:]:
            if g.x - cur[-1].x == TILE:
                cur.append(g)
            else:
                out.append(cur)
                cur = [g]
        out.append(cur)
    return out


@dataclass
class Screen:
    rules: list[tuple[list[Glyph], list[Glyph]]]
    source: list[Glyph]  # the input row
    output: list[Glyph]  # the row with the cursor (the lower row when the rules are edited)
    cursor: int | None  # index into output
    edit_side: int | None = None  # 2 * rule + side, when the cursor is on a rule instead

    def side(self, k: int) -> list[Glyph]:
        return self.rules[k // 2][k % 2]


def _joined(grid: np.ndarray, a: list[Glyph], b: list[Glyph]) -> bool:
    """Is there a line across the gap between sequence ``a`` and ``b`` (on one row)?"""
    x1, x2 = a[-1].x + TILE, b[0].x - 1
    if x2 - x1 < 1:
        return False
    y = a[0].y + TILE // 2
    line = grid[y, x1:x2 + 1]
    if not (line == line[0]).all():
        return False
    above = grid[y - 2, x1:x2 + 1]
    return bool((above != line[0]).all())


def _cursor_on(grid: np.ndarray, seq: list[Glyph], tile_colors: set[int]) -> int | None:
    """The tile of ``seq`` with marks just above and just below it, in a colour no
    tile has and the band round them is not mostly of."""
    h, w = grid.shape
    for i, g in enumerate(seq):
        found = []
        for y1, y2 in ((g.y - 3, g.y - 1), (g.y + TILE + 1, g.y + TILE + 3)):
            if y1 < 0 or y2 > h:
                found.append(set())
                continue
            band = grid[y1:y2, max(g.x - 3, 0):min(g.x + TILE + 3, w)]
            ground = int(np.bincount(band.ravel()).argmax())
            part = grid[y1:y2, g.x + 1:g.x + TILE - 1]
            found.append(set(np.unique(part).tolist()) - tile_colors - {ground})
        if found[0] & found[1]:
            return i
    return None


def read_screen(grid: np.ndarray) -> Screen | None:
    glyphs = find_glyphs(grid)
    if len(glyphs) < 4:
        return None
    seqs = sequences(glyphs)
    rules, used = [], set()
    by_y: dict[int, list[int]] = {}
    for i, s in enumerate(seqs):
        by_y.setdefault(s[0].y, []).append(i)
    for y, idx in by_y.items():
        idx.sort(key=lambda i: seqs[i][0].x)
        for a, b in zip(idx, idx[1:]):
            if a in used or b in used:
                continue
            if _joined(grid, seqs[a], seqs[b]):
                rules.append((seqs[a], seqs[b]))
                used |= {a, b}
    rows = [s for i, s in enumerate(seqs) if i not in used]
    if not rules or len(rows) != 2:
        return None
    tile_colors = set()
    for g in glyphs:
        tile_colors |= set(np.unique(grid[g.y:g.y + TILE, g.x:g.x + TILE]).tolist())
    rules.sort(key=lambda r: (r[0][0].y, r[0][0].x))
    cursor = None
    output = None
    for s in rows:
        c = _cursor_on(grid, s, tile_colors)
        if c is not None:
            output, cursor = s, c
    if output is not None:
        source = rows[0] if rows[1] is output else rows[1]
        return Screen(rules, source, output, cursor)
    # the cursor on a side of a rule: the rules are what is edited, the rows are fixed
    for k in range(2 * len(rules)):
        if _cursor_on(grid, rules[k // 2][k % 2], tile_colors) is not None:
            rows.sort(key=lambda s: s[0].y)
            return Screen(rules, rows[0], rows[1], None, k)
    return None


def translate(screen: Screen) -> list[tuple[int, bytes]] | None:
    """The output the rules make of the input."""
    rules = [([g.id for g in left], [g.id for g in right]) for left, right in screen.rules]
    family = screen.output[0].family if screen.output else None

    def once(seq):
        out, i = [], 0
        while i < len(seq):
            for left, right in rules:
                if seq[i:i + len(left)] == left:
                    out.append(right)
                    i += len(left)
                    break
            else:
                return None
        return out

    def settle(piece, depth=0):
        if all(f == family for f, _ in piece) or depth > 3:
            return piece
        for left, right in rules:
            if left == piece:
                return settle(right, depth + 1)
        parts = once(piece)  # glyph by glyph
        if parts is None:
            return None
        flat = [g for p in parts for g in p]
        return settle(flat, depth + 1) if flat != piece else None

    pieces = once([g.id for g in screen.source])
    if pieces is None:
        return None
    out = []
    for p in pieces:
        q = settle(p)
        if q is None:
            return None
        out += q
    return out


CYCLE = 7  # a family's glyphs, in the order a key turns them


def solve_edit(screen: Screen, shift, start: int, nodes_cap: int = 50000) -> dict[int, int] | None:
    """Turns for each rule side (how many forward) after which the rules rewrite the
    input as the output, cheapest first (turns, then cursor travel). A rule is tried
    where the parse reaches it, in reading order, the first whose left side matches
    applying; a side's turn is fixed the first time the parse needs it, and a rule
    passed over keeps the turn that would have made it match ruled out. A right side
    in another family than the output's is rewritten glyph by glyph by the first rule
    whose left side begins with the glyph (if one does not, the next rule is tried)."""
    sides = [[g.id for g in screen.side(k)] for k in range(2 * len(screen.rules))]
    src = [g.id for g in screen.source]
    out = [g.id for g in screen.output]
    family = out[0][0]
    n_sides = len(sides)
    best: list = [None, None]
    nodes = [0]

    def turned(k, o):
        res = [shift(g, o) for g in sides[k]]
        return None if any(g is None for g in res) else res

    def match_turn(k, want, prefix_only=False):
        """The turn of side k that makes it equal ``want`` (or begin with it)."""
        for o in range(CYCLE):
            t = turned(k, o)
            if t is not None and (t[:len(want)] == want if prefix_only else t == want):
                return o
        return None

    def cost(turns):
        moves = [k for k, o in turns.items() if o % CYCLE]
        clicks = sum(min(o, CYCLE - o) for o in (turns[k] for k in moves))
        here, travel, left = start, 0, set(moves)
        while left:
            k = min(left, key=lambda k: min((k - here) % n_sides, (here - k) % n_sides))
            travel += min((k - here) % n_sides, (here - k) % n_sides)
            here = k
            left.remove(k)
        return clicks + travel

    def finish(turns, forbid):
        full = dict(turns)
        for k in range(n_sides):
            if k not in full:
                allowed = [o for o in range(CYCLE) if o not in forbid.get(k, ())]
                if not allowed:
                    return
                full[k] = min(allowed, key=lambda o: min(o, CYCLE - o))
        c = cost(full)
        if best[0] is None or c < best[0]:
            best[0], best[1] = c, full

    def with_turn(turns, k, o):
        t = dict(turns)
        t[k] = o
        return t

    def with_forbid(forbid, k, o):
        f = {a: set(b) for a, b in forbid.items()}
        f.setdefault(k, set()).add(o)
        return f

    def side_is(turns, forbid, k, want, prefix_only=False):
        """(turns, forbid) choices under which side k equals ``want``, and those under
        which it does not."""
        if k in turns:
            t = turned(k, turns[k])
            ok = t is not None and (t[:len(want)] == want if prefix_only else t == want)
            return ([(turns, forbid)] if ok else []), ([] if ok else [(turns, forbid)])
        o = match_turn(k, want, prefix_only)
        if o is None or o in forbid.get(k, ()):
            return [], [(turns, forbid)]
        return [(with_turn(turns, k, o), forbid)], [(turns, with_forbid(forbid, k, o))]

    def rewrite(glyphs, j, turns, forbid, then, otherwise):
        """Each glyph by the first rule whose left side begins with it; the rights in
        order must be the output from j: ``then(j, turns, forbid)`` after the last
        glyph, ``otherwise(turns, forbid)`` where a glyph has no rule (the caller then
        tries its next rule)."""
        if not glyphs:
            then(j, turns, forbid)
            return
        g = glyphs[0]

        def at(r, turns, forbid):
            if r == len(screen.rules):
                otherwise(turns, forbid)
                return
            yes, no = side_is(turns, forbid, 2 * r, [g], prefix_only=True)
            for t, f in yes:
                need = len(sides[2 * r + 1])
                piece = out[j:j + need]
                if len(piece) == need:
                    for t2, f2 in side_is(t, f, 2 * r + 1, piece)[0]:
                        rewrite(glyphs[1:], j + need, t2, f2, then, otherwise)
            for t, f in no:
                at(r + 1, t, f)

        at(0, turns, forbid)

    def parse(i, j, turns, forbid):
        nodes[0] += 1
        if nodes[0] > nodes_cap:
            return
        if i == len(src):
            if j == len(out):
                finish(turns, forbid)
            return

        def at(r, turns, forbid):
            if r == len(screen.rules) or nodes[0] > nodes_cap:
                return
            left = sides[2 * r]
            seg = src[i:i + len(left)]
            if len(seg) < len(left):
                at(r + 1, turns, forbid)
                return
            yes, no = side_is(turns, forbid, 2 * r, seg)
            for t, f in yes:
                right_k = 2 * r + 1
                if sides[right_k][0][0] == family:
                    piece = out[j:j + len(sides[right_k])]
                    for t2, f2 in side_is(t, f, right_k, piece)[0]:
                        parse(i + len(left), j + len(piece), t2, f2)
                    continue
                for o in range(CYCLE):
                    if (right_k in t and t[right_k] != o) or o in f.get(right_k, ()):
                        continue
                    mid = turned(right_k, o)
                    if mid is None:
                        continue
                    rewrite(mid, j, with_turn(t, right_k, o), f,
                            lambda j2, t3, f3: parse(i + len(left), j2, t3, f3),
                            lambda t3, f3: at(r + 1, t3, f3))
            for t, f in no:
                at(r + 1, t, f)

        at(0, turns, forbid)

    parse(0, 0, {}, {})
    return best[1]


class TranslateSolver:
    name = "translate"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.disabled = False
        self.stats: Counter = Counter()
        self.level = -1
        self.confirmed = False
        self.in_control = False
        # family -> pattern -> the pattern the forward key turns it into (and back)
        self.forward: dict[int, dict[bytes, bytes]] = {}
        self.turn_keys: tuple[GameAction, GameAction] = (GameAction.ACTION2, GameAction.ACTION1)
        self.move_keys: tuple[GameAction, GameAction] = (GameAction.ACTION4, GameAction.ACTION3)
        self._new_level()

    def _new_level(self) -> None:
        self.gave_up = False
        self.fails = 0
        self.level_actions = 0
        self.last: tuple | None = None  # (action, screen before)
        self.read = False
        self.edit_goal: dict[int, list] | None = None  # rule side -> the glyphs it must show

    # -- observing --------------------------------------------------------------

    def observe(self, action: GameAction | None, data: dict | None, obs: FrameDataRaw) -> None:
        if self.disabled or obs.is_empty() or obs.state not in (GameState.NOT_FINISHED, GameState.WIN):
            return
        if obs.levels_completed != self.level:
            if obs.levels_completed > self.level >= 0 and self.in_control:
                self.confirmed = True
                self.stats["levels"] += 1
            self.level = obs.levels_completed
            self.in_control = False
            self._new_level()
            return
        last, self.last = self.last, None
        if last is None or action is None or action is GameAction.RESET:
            return
        act, before = last
        after = read_screen(np.asarray(obs.frame[-1]))
        if after is None:
            return
        if before.edit_side is not None:
            k = before.edit_side
            if act in self.turn_keys:
                p, q = before.side(k)[0], after.side(k)[0] if k < 2 * len(after.rules) else None
                if q is None or p.id == q.id:
                    self.fails += 1
                    return
                links = self.forward.setdefault(p.family, {})
                if act is self.turn_keys[0]:
                    links[p.shape] = q.shape
                else:
                    links[q.shape] = p.shape
            elif act in self.move_keys:
                n = 2 * len(before.rules)
                step = 1 if act is self.move_keys[0] else -1
                if after.edit_side != (k + step) % n:
                    self.fails += 1
                    self.stats["no_move"] += 1
            return
        if before.cursor is None:
            return
        if act in self.turn_keys:
            if after.cursor != before.cursor or len(after.output) != len(before.output):
                self.fails += 1
                return
            p, q = before.output[before.cursor], after.output[after.cursor]
            if p.id == q.id:
                self.fails += 1
                self.stats["no_turn"] += 1
                return
            links = self.forward.setdefault(p.family, {})
            if act is self.turn_keys[0]:
                links[p.shape] = q.shape
            else:
                links[q.shape] = p.shape
        elif act in self.move_keys:
            n = len(before.output)
            step = 1 if act is self.move_keys[0] else -1
            if after.cursor != (before.cursor + step) % n:
                self.fails += 1
                self.stats["no_move"] += 1

    # -- planning ---------------------------------------------------------------

    def _links(self, family: int) -> dict[bytes, bytes]:
        """The forward links of a family; with all but one known, the last closes the cycle."""
        links = self.forward.get(family, {})
        if len(links) == CYCLE - 1:
            tails = set(links) - set(links.values())
            heads = set(links.values()) - set(links)
            if len(tails) == 1 and len(heads) == 1:
                links = dict(links)
                links[heads.pop()] = tails.pop()
        return links

    def _shift(self, gid: tuple[int, bytes], o: int) -> tuple[int, bytes] | None:
        links = self._links(gid[0])
        cur = gid[1]
        for _ in range(o % CYCLE):
            if cur not in links:
                return None
            cur = links[cur]
        return (gid[0], cur)

    def _turns(self, family: int, start: bytes, goal: bytes) -> tuple[int | None, int | None]:
        """Key presses forward and back from ``start`` to ``goal``, where known."""
        links = self._links(family)
        back = {v: k for k, v in links.items()}
        res = []
        for table in (links, back):
            cur, n = start, 0
            while cur != goal and cur in table and n < 12:
                cur = table[cur]
                n += 1
            res.append(n if cur == goal else None)
        return res[0], res[1]

    def _give_up(self, why: str) -> None:
        self.stats["give_up_" + why] += 1
        self.gave_up = True
        self.in_control = False

    def act(self, obs: FrameDataRaw) -> Action | None:
        if self.disabled or self.gave_up or obs.state != GameState.NOT_FINISHED or obs.is_empty():
            return None
        if not {1, 2, 3, 4} <= set(obs.available_actions):
            return None
        screen = read_screen(np.asarray(obs.frame[-1]))
        first, self.read = not self.read, True
        if screen is None or (screen.cursor is None and screen.edit_side is None):
            if first or self.in_control:
                self._give_up("no_screen")
            return None
        if self.fails >= MAX_FAILS or self.level_actions >= MAX_LEVEL_ACTIONS:
            self._give_up("fails" if self.fails >= MAX_FAILS else "actions")
            return None
        if screen.edit_side is not None:
            return self._edit(screen)
        target = translate(screen)
        if target is None or len(target) != len(screen.output):
            self._give_up("no_target")
            return None
        out = [g.id for g in screen.output]
        n = len(out)
        wrong = [i for i in range(n) if out[i] != target[i]]
        if not wrong:
            return None  # done; the game should be moving on
        cur = screen.cursor
        if out[cur] != target[cur]:
            g = screen.output[cur]
            fwd, back = self._turns(g.family, g.shape, target[cur][1])
            if back is not None and (fwd is None or back < fwd):
                action = self.turn_keys[1]
            else:
                action = self.turn_keys[0]
        else:
            # the nearest wrong glyph, round the row either way
            right = min((i - cur) % n for i in wrong)
            left = min((cur - i) % n for i in wrong)
            action = self.move_keys[0] if right <= left else self.move_keys[1]
        return self._take(action, screen)

    def _take(self, action: GameAction, screen: Screen) -> Action:
        if not self.in_control:
            self.in_control = True
            self.stats["takeovers"] += 1
        self.level_actions += 1
        self.stats["actions"] += 1
        self.last = (action, screen)
        return action, None

    def _edit(self, screen: Screen) -> Action | None:
        """Turn the rules' sides until the rules rewrite the input as the output."""
        n = 2 * len(screen.rules)
        if self.edit_goal is None or len(self.edit_goal) != n:
            turns = solve_edit(screen, self._shift, screen.edit_side)
            if turns is None:
                self._give_up("no_edit")
                return None
            goal = {}
            for k, o in turns.items():
                ids = [self._shift(g.id, o) for g in screen.side(k)]
                goal[k] = ids
            self.edit_goal = goal
            self.stats["edit_plans"] += 1
        todo = [k for k in range(n) if [g.id for g in screen.side(k)] != self.edit_goal[k]]
        if not todo:
            return None
        k = screen.edit_side
        if k in todo:
            g = screen.side(k)[0]
            fwd, back = self._turns(g.family, g.shape, self.edit_goal[k][0][1])
            action = self.turn_keys[1] if back is not None and (fwd is None or back < fwd) else self.turn_keys[0]
        else:
            right = min((t - k) % n for t in todo)
            left = min((k - t) % n for t in todo)
            action = self.move_keys[0] if right <= left else self.move_keys[1]
        return self._take(action, screen)
