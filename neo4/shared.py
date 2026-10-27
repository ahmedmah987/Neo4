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


def goal_priors() -> GoalPriors:
    """The process's priors, shared by every agent in it."""
    return _PRIORS


def reset() -> None:
    """Forget everything (between offline experiments)."""
    global _PRIORS
    _PRIORS = GoalPriors()
