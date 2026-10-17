"""Path planning for the avatar inside the induced model.

Breadth-first search over avatar positions on the 64x64 grid. A move is
allowed when every cell the avatar would enter is a colour the model knows
to be passable; the search stops at the first move that would enter a
*goal* cell (a colour that completed a level before). Nothing here touches
the game: the result is a list of action ids for the agent to execute and
verify step by step.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from .objects import Obj

MAX_EXPANSIONS = 20000


def cells_ahead(
    grid: np.ndarray, rel: np.ndarray, pos: tuple[int, int], d: tuple[int, int]
) -> set[int] | None:
    """Colours of the cells a sprite with pixel offsets ``rel`` enters when moving
    from top-left ``pos`` by ``d``; None when the move leaves the grid."""
    x, y = pos
    dx, dy = d
    ys = rel[:, 0] + y + dy
    xs = rel[:, 1] + x + dx
    if ys.min() < 0 or xs.min() < 0 or ys.max() >= grid.shape[0] or xs.max() >= grid.shape[1]:
        return None
    own = set(zip((rel[:, 0] + y).tolist(), (rel[:, 1] + x).tolist()))
    return {int(grid[yy, xx]) for yy, xx in zip(ys.tolist(), xs.tolist()) if (yy, xx) not in own}


def plan_path(
    grid: np.ndarray,
    me: Obj,
    dirs: dict[int, tuple[int, int]],
    passable: Callable[[int], bool],
    is_goal: Callable[[set[int]], bool],
) -> tuple[list[int], list[tuple[int, int]]] | None:
    """Actions (and the avatar position expected after each) to reach a goal.

    The last action is the one that enters the goal; the position recorded for
    it is where the avatar would land if the move went through.
    """
    rel = me.pixels - np.array([me.y1, me.x1])
    start = (me.x1, me.y1)
    parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    expansions = 0

    def unwind(pos: tuple[int, int]) -> tuple[list[int], list[tuple[int, int]]]:
        actions: list[int] = []
        positions: list[tuple[int, int]] = []
        while pos != start:
            prev, aid = parent[pos]
            actions.append(aid)
            positions.append(pos)
            pos = prev
        return actions[::-1], positions[::-1]

    while queue and expansions < MAX_EXPANSIONS:
        pos = queue.popleft()
        expansions += 1
        for aid, d in dirs.items():
            colors = cells_ahead(grid, rel, pos, d)
            if colors is None:
                continue
            nxt = (pos[0] + d[0], pos[1] + d[1])
            if is_goal(colors):
                actions, positions = unwind(pos)
                return actions + [aid], positions + [nxt]
            if nxt in seen or not all(passable(c) for c in colors):
                continue
            seen.add(nxt)
            parent[nxt] = (pos, aid)
            queue.append(nxt)
    return None
