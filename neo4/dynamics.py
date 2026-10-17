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
DEATH_SUPPORT = 2  # lost games on a click target before its clicks count as dangerous
NOOP_RATE = 0.95  # share of clicks on a signature that changed nothing
DIR_PURITY = 0.7  # share of an action's observed moves that agree on one direction
GOAL_LIFT = 5.0  # a goal feature ends levels at least this many times the base rate
NOISE_RATE = 0.5  # a signature churning on this share of steps under every action is noise
NOISE_MIN_STEPS = 10  # observations an action class needs before it counts for noise
NOISE_REFRESH = 25  # recompute the noise set every this many observations


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
        self.goal_ahead: Counter = Counter()  # colours ahead of the avatar when a level ended
        self.goal_click: Counter = Counter()  # signatures clicked when a level ended
        self.goal_click_color: Counter = Counter()  # colours clicked when a level ended
        self.click_color: Counter = Counter()  # colours clicked, all outcomes
        self.level_ups = 0
        self.click_death: Counter = Counter()  # signatures clicked when the game was lost
        # The same, per object instance (signature + position): outcomes often depend on
        # *which* of several identical objects is clicked, not on their kind.
        self.click_at: dict[tuple, Counter] = defaultdict(Counter)
        self.click_death_at: Counter = Counter()
        self.churn: dict[tuple, Counter] = defaultdict(Counter)  # sig -> action class -> events
        self.touched: set[tuple] = set()  # object signatures the avatar has moved into / against
        self.steps_by_class: Counter = Counter()  # action class -> transitions observed
        self.observations = 0
        self._avatar: tuple | None = None
        self._avatar_dirty = True
        self._noise: frozenset = frozenset()
        self._noise_at = 0

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
        if outcome == LEVEL_UP:
            self.level_ups += 1
        d = diff(before, after) if after is not None and outcome == OK else None
        if d is not None:
            # Clicks on different targets are different actions: something that
            # changes only when one particular target is clicked is a mechanic,
            # something that changes whatever is clicked is noise.
            if action.is_complex():
                target = at(before, data["x"], data["y"]) if data else None
                cls: object = ("click", target.sig if target is not None else None)
            else:
                cls = action.value
            self.steps_by_class[cls] += 1
            for b, _, _, _ in d.moved:
                self.churn[b.sig][cls] += 1
            for o in d.vanished:
                self.churn[o.sig][cls] += 1
            for o in d.appeared:
                self.churn[o.sig][cls] += 1
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
        key = "change" if changed else "none"
        self.click[target.sig][key] += 1
        self.click_at[(target.sig, target.x1, target.y1)][key] += 1
        self.click_color[target.color] += 1
        if outcome == LEVEL_UP:
            self.goal_click[target.sig] += 1
            self.goal_click_color[target.color] += 1
        elif outcome == GAME_OVER:
            self.click_death[target.sig] += 1
            self.click_death_at[(target.sig, target.x1, target.y1)] += 1

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
        colors, coords = self._ahead_cells(me, direction, before_grid)
        for y, x in coords:
            o = at(before, x, y)
            if o is not None and o.sig != avatar and not o.scenery:
                self.touched.add(o.sig)
        if outcome == GAME_OVER:
            for c in colors:
                self.hazard[c] += 1
            return
        if outcome == LEVEL_UP:
            for c in colors:
                self.goal_ahead[c] += 1
            return
        moved = d is not None and any(b.sig == avatar for b, _, _, _ in d.moved)
        key = "passed" if moved else "blocked"
        for c in colors:
            self.cell[c][key] += 1

    @staticmethod
    def _ahead_cells(
        me: Obj, direction: tuple[int, int], grid: np.ndarray
    ) -> tuple[set[int], list[tuple[int, int]]]:
        """Colours and (y, x) cells the avatar would occupy after the move, minus its own.

        A move off the grid reports the pseudo-colour -1 and no cells."""
        dx, dy = direction
        ys = me.pixels[:, 0] + dy
        xs = me.pixels[:, 1] + dx
        inside = (ys >= 0) & (ys < grid.shape[0]) & (xs >= 0) & (xs < grid.shape[1])
        if not inside.all():
            return {-1}, []
        own = set(zip(me.pixels[:, 0].tolist(), me.pixels[:, 1].tolist()))
        coords = [(y, x) for y, x in zip(ys.tolist(), xs.tolist()) if (y, x) not in own]
        return {int(grid[y, x]) for y, x in coords}, coords

    def _ahead(self, me: Obj, direction: tuple[int, int], grid: np.ndarray) -> set[int]:
        return self._ahead_cells(me, direction, grid)[0]

    # -- derived rules --------------------------------------------------------

    def noise(self) -> frozenset:
        """Signatures that change whatever the agent does: particles, timers, animations.

        A signature is noise when it moved, appeared or vanished on at least
        ``NOISE_RATE`` of the transitions under *every* action class that has
        enough observations. Something that reacts to one particular action
        is not noise; something that churns regardless of the action is.
        """
        if self.observations - self._noise_at >= NOISE_REFRESH or self._noise_at == 0:
            self._noise_at = self.observations
            classes = [c for c, n in self.steps_by_class.items() if n >= NOISE_MIN_STEPS]
            out = set(self._noise)  # membership is sticky: a changing set would re-key the graph
            if len(classes) >= 2:
                for sig, c in self.churn.items():
                    if self._controlled(sig):
                        continue
                    if all(c[cls] / self.steps_by_class[cls] >= NOISE_RATE for cls in classes):
                        out.add(sig)
                # Particles clump into rarer shapes of the same colour; take those along.
                colors = {sig[0] for sig in out}
                for sig in self.churn:
                    if sig[0] in colors and sig not in out and not self._controlled(sig):
                        out.add(sig)
            self._noise = frozenset(out)
        return self._noise

    def _controlled(self, sig: tuple) -> bool:
        """Some keyboard action moves this signature in one consistent direction."""
        for (_, s), c in self.move.items():
            if s != sig:
                continue
            nonzero = [(k, n) for k, n in c.items() if k != (0, 0)]
            if not nonzero:
                continue
            best = max(n for _, n in nonzero)
            total = sum(n for _, n in nonzero)
            if best >= MIN_SUPPORT and best / total >= DIR_PURITY:
                return True
        return False

    def avatar(self) -> tuple | None:
        """Signature that the keyboard moves most often."""
        if self._avatar_dirty:
            noise = self.noise()
            score: Counter = Counter()
            for (_, sig), c in self.move.items():
                if sig in noise:
                    continue
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

    def directions(self) -> dict[int, tuple[int, int]]:
        """Action id -> avatar displacement, for every action with a stable effect."""
        avatar = self.avatar()
        if avatar is None:
            return {}
        out = {}
        for aid in {a for a, s in self.move if s == avatar}:
            d = self.direction(aid)
            if d is not None:
                out[aid] = d
        return out

    def passable(self, color: int) -> bool:
        """The avatar crossed this colour reliably and never died entering it."""
        visits = self.hazard[color] + sum(self.cell[color].values())
        if visits >= MIN_SUPPORT and self.hazard[color] / visits >= DANGER_RATE:
            return False
        passed, n = self._rate(self.cell[color], "passed")
        return n >= MIN_SUPPORT and passed >= PASS_RATE

    def _lifted(self, goals: Counter, visits) -> set:
        """Features under which a level ended far more often than the base rate.

        A feature (a colour ahead, a click target) is a goal when the share of
        its visits that completed a level is at least ``GOAL_LIFT`` times the
        share of all transitions that did. One event on a rare feature counts;
        one event on something seen thousands of times does not.
        """
        if not self.level_ups or not self.observations:
            return set()
        base = self.level_ups / self.observations
        out = set()
        for f, n in goals.items():
            total = visits(f)
            if total and n / total >= GOAL_LIFT * base:
                out.add(f)
        return out

    def goal_colors(self) -> set[int]:
        """Colours ahead of the avatar when a level ended, beyond chance."""
        return self._lifted(
            self.goal_ahead,
            lambda c: self.goal_ahead[c] + sum(self.cell[c].values()) + self.hazard[c],
        )

    def goal_click_sigs(self) -> set[tuple]:
        return self._lifted(self.goal_click, lambda sig: sum(self.click[sig].values()))

    def goal_click_colors(self) -> set[int]:
        return self._lifted(self.goal_click_color, lambda c: self.click_color[c])

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
            inst = (target.sig, target.x1, target.y1)
            # Danger generalises per instance (this cell kills) and per kind; a no-op
            # rule only per kind: a cell that did nothing so far may matter later, and
            # skipping it per instance cost levels on r11l and su15.
            for deaths, counter, where in (
                (self.click_death_at[inst], self.click_at[inst], f"{target.label()} at {target.x1},{target.y1}"),
                (self.click_death[target.sig], self.click[target.sig], target.label()),
            ):
                n = sum(counter.values())
                if deaths >= DEATH_SUPPORT and deaths / n >= DANGER_RATE:
                    return Prediction("danger", n, f"clicking {where} lost the game {deaths}/{n}x")
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
        for sig in sorted(self.noise()):
            total = sum(self.churn[sig].values())
            lines.append(
                f"noise: colour-{sig[0]} {sig[2]}x{sig[1]} changes whatever the action ({total} events)"
            )
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
            for color in sorted(self.goal_colors()):
                lines.append(f"  entering colour {color} completed a level ({self.goal_ahead[color]}x)")
        for sig in self.goal_click_sigs():
            lines.append(
                f"clicking colour-{sig[0]} {sig[2]}x{sig[1]} completed a level ({self.goal_click[sig]}x)"
            )
        for c in sorted(self.goal_click_colors()):
            lines.append(
                f"clicking colour {c} completed a level ({self.goal_click_color[c]}/{self.click_color[c]} clicks)"
            )
        for sig, n in self.click_death.most_common(6):
            total = sum(self.click[sig].values())
            lines.append(f"clicking colour-{sig[0]} {sig[2]}x{sig[1]} lost the game {n}/{total} times")
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
