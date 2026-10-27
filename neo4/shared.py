"""What the goal turned out to be in the other games played in this process (Phase 65).

Every game used to start from nothing: what the agent had learned in one game was gone in the
next, though games that differ in their goals share situations. Here the agent keeps, across
games, a count of what the goals were like: at each level-up, every hypothesis on the screen
adds its features (its kind, whether the keys move an avatar, how its object looks beside the
others and the avatar) to ``seen``, and those that ended the level to ``ended``. A new game's
tests are then ordered by how often guesses like them were goals elsewhere.

Only counts are kept, never a game's layout or its solution; in the competition the games are
played at once, one thread each, so the counts fill as the run goes on.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Iterable

MIN_LEVELS = 3  # level-ups recorded before the counts order anything


class GoalPriors:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.seen: Counter = Counter()
        self.ended: Counter = Counter()
        self.levels = 0
        self.seeded = False

    def seed(self, counts: dict, exclude: str | None = None) -> None:
        """Start from what the goals were like in the public games (Phase 67): ``counts`` maps a
        game to its (levels, seen, ended) at the level-ups the full library reaches there; a game
        named by ``exclude`` is left out, so that a public game played as an unseen one does not
        learn from itself."""
        with self._lock:
            for game, (levels, seen, ended) in counts.items():
                if game == exclude:
                    continue
                self.levels += levels
                for f, n in seen:
                    self.seen[f] += n
                for f, n in ended:
                    self.ended[f] += n
            self.seeded = True

    def record(self, hyps: Iterable, ended: set) -> None:
        """A level ended: ``hyps`` were on the screen, those with a key in ``ended`` ended it."""
        with self._lock:
            self.levels += 1
            for h in hyps:
                for f in h.features:
                    self.seen[f] += 1
                    if h.key in ended:
                        self.ended[f] += 1

    def score(self, features: Iterable) -> float | None:
        """How much guesses like this one were goals elsewhere: the sum over its features of the
        log odds that a guess with the feature ended its level (0 for a feature never seen);
        None until enough levels have ended."""
        with self._lock:
            if self.levels < MIN_LEVELS:
                return None
            total = 0.0
            for f in features:
                e, n = self.ended[f], self.seen[f]
                total += math.log((e + 1) / (n - e + 1))
            return total


_PRIORS = GoalPriors()


def seed_from_public(exclude: str | None = None) -> None:
    """Seed the process's priors from the public games' counts, once."""
    from .priors_data import COUNTS

    if not _PRIORS.seeded:
        _PRIORS.seed(COUNTS, exclude)


def goal_priors() -> GoalPriors:
    """The process's priors, shared by every agent in it."""
    return _PRIORS


def reset() -> None:
    """Forget everything (between offline experiments)."""
    global _PRIORS, _SITUATIONS
    _PRIORS = GoalPriors()
    _SITUATIONS = SituationLibrary()


def similarity(a: Iterable, b: Iterable) -> float:
    """The share of features two hypotheses have in common (Jaccard)."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa or sb else 0.0


class SituationLibrary:
    """Situations seen to end a level, in any game of the process (Phase 66): a hypothesis met in
    the life that ended a level, before the end, and the hypothesis that explained the end ("the
    key, then the door"), both in features any game shares. When a test is met and the level goes
    on, the library says what, met first, made goals like it end a level elsewhere."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pairs: Counter = Counter()  # (precondition features, goal features) -> times seen

    def record(self, before: Iterable, goal: Iterable) -> None:
        with self._lock:
            self.pairs[(frozenset(before), frozenset(goal))] += 1

    def preconditions(self, goal: Iterable, min_similarity: float = 0.5) -> list[tuple[frozenset, float]]:
        """What came first in situations whose goal was like ``goal``, weighted by how often they
        were seen and how alike the goals are."""
        with self._lock:
            out = []
            for (pre, g), n in self.pairs.items():
                s = similarity(g, goal)
                if s >= min_similarity:
                    out.append((pre, n * s))
            return out


_SITUATIONS = SituationLibrary()


def situations() -> SituationLibrary:
    """The process's library of situations, shared by every agent in it."""
    return _SITUATIONS
