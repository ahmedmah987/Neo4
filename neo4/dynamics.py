"""Rule induction: what each action does, learned from recorded transitions.

``Dynamics`` keeps per-game statistics that survive level changes:

* how far each object signature moves under each keyboard action,
* which object is the *avatar* (the one the keyboard moves),
* for each colour, how often it blocked the avatar versus let it pass,
* which colours were in front of the avatar when the game was lost,
* how often clicking a given object signature changed anything.

``predict`` turns those statistics into a verdict on a candidate action in a
given frame — ``unknown``, ``move``, ``blocked``, ``danger`` or
``noop_click`` — with the evidence count behind it, and ``describe`` renders
the learned rules in English.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from arcengine import GameAction

from .objects import Diff, Obj, at, diff, find

OK, GAME_OVER, LEVEL_UP = "ok", "game_over", "level_up"

MIN_SUPPORT = 3  # observations before a rule may be used for a prediction
BLOCK_RATE = 0.9  # a colour that stopped the avatar at least this often blocks
PASS_RATE = 0.9  # a colour the avatar crossed at least this often is passable
DANGER_RATE = 0.5  # share of visits that ended the game
NOOP_RATE = 0.95  # share of clicks on a signature that changed nothing
DIR_PURITY = 0.7  # share of an action's observed moves that agree on one direction


@dataclass(slots=True)
class Prediction:
    verdict: str  # unknown | move | blocked | danger | noop_click
    support: int = 0
    why: str = ""


class Dynamics:
    def __init__(self) -> None:
        self.move: dict[tuple[int, tuple], Counter] = defaultdict(Counter)
        self.click: dict[tuple, Counter] = defaultdict(Counter)
        self.cell: dict[int, Counter] = defaultdict(Counter)
        self.hazard: Counter = Counter()
        self.observations = 0
        self._avatar: tuple | None = None
        self._avatar_dirty = True

    # -- learning -------------------------------------------------------------

    def observe(
        self,
        action: GameAction,
        data: dict[str, int] | None,
        before: list[Obj],
        before_grid: np.ndarray,
        after: list[Obj] | None,
        outcome: str,
    ) -> None:
        self.observations += 1
        d = diff(before, after) if after is not None and outcome == OK else None
        if action.is_complex():
            self._observe_click(data, before, d, outcome)
        else:
            self._observe_key(action, before, before_grid, d, outcome)

    def _observe_click(self, data, before, d: Diff | None, outcome: str) -> None:
        if data is None:
            return
        target = at(before, data["x"], data["y"])
        if target is None:
            return
        changed = outcome != OK or (d is not None and d.changed)
        self.click[target.sig]["change" if changed else "none"] += 1

    def _observe_key(self, action, before, before_grid, d: Diff | None, outcome: str) -> None:
        aid = action.value
        if d is not None:
            for b, _, dx, dy in d.moved:
                self.move[(aid, b.sig)][(dx, dy)] += 1
                self._avatar_dirty = True
            for b in d.unchanged:
                if (aid, b.sig) in self.move:
                    self.move[(aid, b.sig)][(0, 0)] += 1
        avatar = self.avatar()
        if avatar is None:
            return
        direction = self.direction(aid)
        if direction is None:
            return
        me = find(before, avatar)
        if me is None:
            return
        colors = self._ahead(me, direction, before_grid)
        if outcome == GAME_OVER:
            for c in colors:
                self.hazard[c] += 1
            return
        moved = d is not None and any(b.sig == avatar for b, _, _, _ in d.moved)
        key = "passed" if moved else "blocked"
        for c in colors:
            self.cell[c][key] += 1

    @staticmethod
    def _ahead(me: Obj, direction: tuple[int, int], grid: np.ndarray) -> set[int]:
        """Colours of the cells the avatar would occupy after the move, minus its own."""
        dx, dy = direction
        ys = me.pixels[:, 0] + dy
        xs = me.pixels[:, 1] + dx
        inside = (ys >= 0) & (ys < grid.shape[0]) & (xs >= 0) & (xs < grid.shape[1])
        if not inside.all():
            return {-1}  # the grid edge
        own = set(zip(me.pixels[:, 0].tolist(), me.pixels[:, 1].tolist()))
        return {
            int(grid[y, x]) for y, x in zip(ys.tolist(), xs.tolist()) if (y, x) not in own
        }

    # -- derived rules --------------------------------------------------------

    def avatar(self) -> tuple | None:
        """Signature that the keyboard moves most often."""
        if self._avatar_dirty:
            score: Counter = Counter()
            for (_, sig), c in self.move.items():
                score[sig] += sum(n for (dx, dy), n in c.items() if (dx, dy) != (0, 0))
            # Ties (a hole inside the avatar moves exactly as often) go to the bigger sprite.
            best = max(score.items(), key=lambda kv: (kv[1], kv[0][1] * kv[0][2]), default=None)
            self._avatar = best[0] if best and best[1] >= MIN_SUPPORT else None
            self._avatar_dirty = False
        return self._avatar

    def direction(self, action_id: int) -> tuple[int, int] | None:
        avatar = self.avatar()
        if avatar is None:
            return None
        c = self.move.get((action_id, avatar))
        if not c:
            return None
        nonzero = [(k, n) for k, n in c.items() if k != (0, 0)]
        if not nonzero:
            return None
        (dx, dy), n = max(nonzero, key=lambda kv: kv[1])
        total = sum(m for _, m in nonzero)
        if n < MIN_SUPPORT or n / total < DIR_PURITY:
            return None
        return dx, dy

    def _rate(self, counter: Counter, key: str) -> tuple[float, int]:
        total = sum(counter.values())
        return (counter[key] / total if total else 0.0), total

    # -- prediction -----------------------------------------------------------

    def predict(
        self, action: GameAction, data: dict[str, int] | None, objs: list[Obj], grid: np.ndarray
    ) -> Prediction:
        if action.is_complex():
            if data is None:
                return Prediction("unknown")
            target = at(objs, data["x"], data["y"])
            if target is None:
                return Prediction("unknown")
            rate, n = self._rate(self.click[target.sig], "none")
            if n >= MIN_SUPPORT and rate >= NOOP_RATE:
                return Prediction("noop_click", n, f"clicking {target.label()} changed nothing {n}x")
            return Prediction("unknown", n)

        avatar = self.avatar()
        direction = self.direction(action.value)
        if avatar is None or direction is None:
            return Prediction("unknown")
        me = find(objs, avatar)
        if me is None:
            return Prediction("unknown")
        colors = self._ahead(me, direction, grid)
        verdict_danger = verdict_block = None
        all_pass = True
        support = 0
        for c in colors:
            visits = self.hazard[c] + sum(self.cell[c].values())
            if visits >= MIN_SUPPORT and self.hazard[c] / visits >= DANGER_RATE:
                verdict_danger = (c, visits)
            blocked, n = self._rate(self.cell[c], "blocked")
            if n >= MIN_SUPPORT and blocked >= BLOCK_RATE:
                verdict_block = (c, n)
            passed, n = self._rate(self.cell[c], "passed")
            if not (n >= MIN_SUPPORT and passed >= PASS_RATE):
                all_pass = False
            support += n
        if verdict_danger:
            c, n = verdict_danger
            return Prediction("danger", n, f"colour {c} ahead ended the game before")
        if verdict_block:
            c, n = verdict_block
            return Prediction("blocked", n, f"colour {c} ahead blocks the avatar")
        if all_pass and colors:
            return Prediction("move", support, f"avatar moves {direction}")
        return Prediction("unknown", support)

    # -- reporting ------------------------------------------------------------

    def describe(self) -> list[str]:
        lines: list[str] = [f"{self.observations} transitions observed"]
        avatar = self.avatar()
        if avatar is None:
            lines.append("no avatar found (nothing moves consistently under the keyboard)")
        else:
            lines.append(f"avatar: colour-{avatar[0]} {avatar[2]}x{avatar[1]} sprite")
            for aid in sorted({a for a, s in self.move if s == avatar}):
                c = self.move[(aid, avatar)]
                d = self.direction(aid)
                total = sum(c.values())
                if d is None:
                    lines.append(f"  ACTION{aid}: no consistent effect ({dict(c)})")
                else:
                    stuck = c[(0, 0)]
                    lines.append(
                        f"  ACTION{aid}: moves it by {d} ({total - stuck}/{total} times; "
                        f"stuck {stuck}x)"
                    )
            for color, c in sorted(self.cell.items()):
                blocked, n = self._rate(c, "blocked")
                if n >= MIN_SUPPORT:
                    what = "edge" if color == -1 else f"colour {color}"
                    if blocked >= BLOCK_RATE:
                        lines.append(f"  {what} blocks the avatar ({c['blocked']}/{n})")
                    elif 1 - blocked >= PASS_RATE:
                        lines.append(f"  {what} is walkable ({c['passed']}/{n})")
                    else:
                        lines.append(
                            f"  {what} ahead: blocked {c['blocked']}/{n} times "
                            "(depends on something else)"
                        )
            for color, n in self.hazard.most_common():
                visits = n + sum(self.cell[color].values())
                lines.append(f"  colour {color} ahead ended the game {n}/{visits} times")
        other = [
            (aid, sig, c)
            for (aid, sig), c in self.move.items()
            if sig != avatar and sum(n for k, n in c.items() if k != (0, 0)) >= MIN_SUPPORT
        ]
        for aid, sig, c in sorted(other, key=lambda t: -sum(t[2].values()))[:8]:
            lines.append(f"ACTION{aid} also moves colour-{sig[0]} {sig[2]}x{sig[1]}: {dict(c)}")
        noops = [(sig, c) for sig, c in self.click.items() if sum(c.values()) >= MIN_SUPPORT]
        for sig, c in sorted(noops, key=lambda t: -sum(t[1].values()))[:12]:
            total = sum(c.values())
            lines.append(
                f"click on colour-{sig[0]} {sig[2]}x{sig[1]}: changed something "
                f"{c['change']}/{total} times"
            )
        return lines
