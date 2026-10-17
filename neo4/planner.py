"""Path planning for the avatar inside the induced model.

Breadth-first search over avatar positions on the 64x64 grid. A move is
allowed when every cell the avatar would pass through is a colour the model
knows to be passable; the search stops at the first move that would enter a
*goal* cell (a colour that completed a level before). Nothing here touches
the game: the result is a list of action ids for the agent to execute and
verify step by step.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from .objects import Obj

MAX_EXPANSIONS = 60000


def swept(
    grid: np.ndarray, rel: np.ndarray, pos: tuple[int, int], d: tuple[int, int]
) -> list[tuple[set[int], list[tuple[int, int]]]] | None:
    """The cells a sprite with pixel offsets ``rel`` passes through when moving
    from top-left ``pos`` by ``d``, as bands: one (colours, (y, x) cells) entry
    per unit step along the move, nearest first, each cell listed once in the
    band where the sprite first enters it. Cells the sprite already occupies
    are left out. None when the move leaves the grid.

    A sprite that jumps several pixels per action (tu93 moves six) can be
    stopped by a wall it never lands on; the bands say what is in the way.
    """
    x, y = pos
    dx, dy = d
    n = max(abs(dx), abs(dy))
    if n == 0:
        return []
    ys = rel[:, 0] + y + dy
    xs = rel[:, 1] + x + dx
    if ys.min() < 0 or xs.min() < 0 or ys.max() >= grid.shape[0] or xs.max() >= grid.shape[1]:
        return None
    seen = set(zip((rel[:, 0] + y).tolist(), (rel[:, 1] + x).tolist()))
    bands: list[tuple[set[int], list[tuple[int, int]]]] = []
    for k in range(1, n + 1):
        ox, oy = round(dx * k / n), round(dy * k / n)
        coords = []
        for yy, xx in zip((rel[:, 0] + y + oy).tolist(), (rel[:, 1] + x + ox).tolist()):
            if (yy, xx) not in seen:
                seen.add((yy, xx))
                coords.append((yy, xx))
        bands.append(({int(grid[yy, xx]) for yy, xx in coords}, coords))
    return bands


def ahead(
    grid: np.ndarray, rel: np.ndarray, pos: tuple[int, int], d: tuple[int, int]
) -> tuple[set[int], list[tuple[int, int]]] | None:
    """Colours and (y, x) coordinates of every cell a sprite with pixel offsets
    ``rel`` passes through when moving from top-left ``pos`` by ``d`` (the union
    of the bands of ``swept``); None when the move leaves the grid."""
    bands = swept(grid, rel, pos, d)
    if bands is None:
        return None
    colors: set[int] = set()
    coords: list[tuple[int, int]] = []
    for c, cells in bands:
        colors |= c
        coords.extend(cells)
    return colors, coords


def cells_ahead(
    grid: np.ndarray, rel: np.ndarray, pos: tuple[int, int], d: tuple[int, int]
) -> set[int] | None:
    """Colours of the cells entered by the move (see ``ahead``)."""
    r = ahead(grid, rel, pos, d)
    return None if r is None else r[0]


def plan_path(
    grid: np.ndarray,
    me: Obj,
    dirs: dict[int, tuple[int, int]],
    passable: Callable[[int], bool],
    is_goal: Callable[[set[int], list[tuple[int, int]]], bool],
    avoid: np.ndarray | None = None,
    forbidden: Callable[[tuple[int, int], int], bool] | None = None,
    avoid_t: list[np.ndarray] | None = None,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    """Actions (and the avatar position expected after each) to reach a goal.

    ``is_goal`` sees the colours and the coordinates of the cells a move would
    enter. The last action is the one that enters the goal; the position
    recorded for it is where the avatar would land if the move went through.
    ``avoid`` marks cells no move may enter (a patrol's reach); ``forbidden``
    rejects a (position, action) pair outright (a move that lost the game);
    ``avoid_t[t]`` marks the cells a moving hazard is predicted to reach after
    ``t + 1`` moves, so the search runs over (position, time) while the
    prediction lasts and over positions alone beyond it.
    """
    rel = me.pixels - np.array([me.y1, me.x1])
    start = (me.x1, me.y1)
    horizon = len(avoid_t) if avoid_t else 0
    State = tuple[tuple[int, int], int]
    parent: dict[State, tuple[State, int]] = {}
    seen = {(start, 0)}
    queue: deque[State] = deque([(start, 0)])
    expansions = 0

    def unwind(state: State) -> tuple[list[int], list[tuple[int, int]]]:
        actions: list[int] = []
        positions: list[tuple[int, int]] = []
        while state != (start, 0):
            prev, aid = parent[state]
            actions.append(aid)
            positions.append(state[0])
            state = prev
        return actions[::-1], positions[::-1]

    while queue and expansions < MAX_EXPANSIONS:
        state = queue.popleft()
        pos, t = state
        expansions += 1
        for aid, d in dirs.items():
            if forbidden is not None and forbidden(pos, aid):
                continue
            r = ahead(grid, rel, pos, d)
            if r is None:
                continue
            colors, coords = r
            if avoid is not None and any(avoid[y, x] for y, x in coords):
                continue
            if t < horizon and any(avoid_t[t][y, x] for y, x in coords):  # type: ignore[index]
                continue
            nxt = (pos[0] + d[0], pos[1] + d[1])
            if is_goal(colors, coords):
                actions, positions = unwind(state)
                return actions + [aid], positions + [nxt]
            nstate = (nxt, min(t + 1, horizon))
            if nstate in seen or not all(passable(c) for c in colors):
                continue
            seen.add(nstate)
            parent[nstate] = (state, aid)
            queue.append(nstate)
    return None
