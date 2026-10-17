"""Frame processing for ARC-AGI-3 observations.

An observation is a 64x64 grid of palette indices 0..15. This module turns it
into the pieces the exploration agent needs:

* connected components of a single colour (``segment``),
* a mask of pixels that are probably a status bar — step counters, energy
  bars — so that they do not leak into the state identity (``status_bar_mask``),
* a stable state key for the masked frame (``state_key``),
* click candidates, one per component, bucketed into priority groups
  (``click_groups``).

The rules follow the baseline of Rudakov, Shock & Cowley (arXiv:2512.24156):
4-connectivity, status bars detected by position at the grid edge plus either
a long aspect ratio or a row of identical "dots", and five click priority
tiers driven by colour salience and component size.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

GRID = 64
PALETTE_SIZE = 16
MASK_COLOR = 16  # outside the palette: marks status-bar pixels before segmentation

# Status-bar rules.
EDGE_DISTANCE = 3  # a component touches an edge if it lies within this many pixels of it
BAR_RATIO = 5.0  # long side / short side at or above this -> bar
DOT_BAR_MIN_MEMBERS = 3  # a component plus its edge twins, for "dot" style bars

# Click priority rules.
SALIENT_MIN_COLOR = 6  # colours 6..15 count as salient, 0..5 as background-like
MEDIUM_MIN = 2  # bounding-box side lengths in [MEDIUM_MIN, MEDIUM_MAX] are "button sized"
MEDIUM_MAX = 32
N_GROUPS = 5

_STRUCT_4 = ndimage.generate_binary_structure(2, 1)
_STRUCT_8 = ndimage.generate_binary_structure(2, 2)


@dataclass(slots=True)
class Component:
    """A single-colour 4-connected region of the frame."""

    id: int
    color: int
    pixels: np.ndarray  # shape (n, 2): rows of (y, x)
    x1: int
    y1: int
    x2: int  # inclusive
    y2: int  # inclusive

    @property
    def area(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def is_rectangle(self) -> bool:
        return self.area == self.width * self.height

    def click_point(self) -> tuple[int, int]:
        """Pixel of the component closest to its bounding-box centre, as (x, y).

        Deterministic, unlike the baseline's random pixel, so runs are
        reproducible for a given seed.
        """
        cy = (self.y1 + self.y2) / 2.0
        cx = (self.x1 + self.x2) / 2.0
        d = (self.pixels[:, 0] - cy) ** 2 + (self.pixels[:, 1] - cx) ** 2
        y, x = self.pixels[int(np.argmin(d))]
        return int(x), int(y)


def segment(frame: np.ndarray, connectivity: int = 4) -> list[Component]:
    """Split a frame into single-colour connected components."""
    if frame.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {frame.shape}")
    struct = _STRUCT_4 if connectivity == 4 else _STRUCT_8
    labels = np.zeros(frame.shape, dtype=np.int32)
    colors: list[int] = []
    offset = 0
    for color in np.unique(frame):
        lab, n = ndimage.label(frame == color, structure=struct)
        if n == 0:
            continue
        mask = lab > 0
        labels[mask] = lab[mask] + offset
        colors.extend([int(color)] * n)
        offset += n

    # Group pixel coordinates by label in one pass.
    flat = labels.ravel() - 1
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=offset)
    ys, xs = np.divmod(order, frame.shape[1])
    comps: list[Component] = []
    start = 0
    for cid in range(offset):
        n = int(counts[cid])
        py = ys[start : start + n]
        px = xs[start : start + n]
        start += n
        comps.append(
            Component(
                id=cid,
                color=colors[cid],
                pixels=np.stack([py, px], axis=1),
                x1=int(px.min()),
                y1=int(py.min()),
                x2=int(px.max()),
                y2=int(py.max()),
            )
        )
    return comps


def _edge_orientation(c: Component, grid: int) -> str | None:
    """'horizontal', 'vertical', 'any' or None depending on which edges the component hugs."""
    left = c.x2 < EDGE_DISTANCE
    right = c.x1 > grid - 1 - EDGE_DISTANCE
    top = c.y2 < EDGE_DISTANCE
    bottom = c.y1 > grid - 1 - EDGE_DISTANCE
    horizontal = top or bottom
    vertical = left or right
    if horizontal and vertical:
        return "any"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    return None


def _is_long_bar(c: Component, orientation: str) -> bool:
    ratio = c.width / c.height
    if orientation in ("horizontal", "any") and ratio >= BAR_RATIO:
        return True
    if orientation in ("vertical", "any") and ratio <= 1.0 / BAR_RATIO:
        return True
    return False


def status_bar_mask(components: list[Component], grid: int = GRID) -> np.ndarray:
    """Boolean mask of pixels that probably belong to a status bar.

    A component is a status bar when it hugs an edge of the grid and is either
    a long thin bar, or one of at least ``DOT_BAR_MIN_MEMBERS`` identical
    "dots" (same colour, area and shape) that also hug an edge.
    """
    mask = np.zeros((grid, grid), dtype=bool)
    edge = {c.id: _edge_orientation(c, grid) for c in components}
    edge_comps = [c for c in components if edge[c.id] is not None]
    for c in edge_comps:
        orientation = edge[c.id]
        assert orientation is not None
        if _is_long_bar(c, orientation):
            mask[c.pixels[:, 0], c.pixels[:, 1]] = True
            continue
        twins = [
            t
            for t in edge_comps
            if t.id != c.id
            and t.color == c.color
            and t.area == c.area
            and t.is_rectangle == c.is_rectangle
        ]
        if len(twins) + 1 >= DOT_BAR_MIN_MEMBERS:
            mask[c.pixels[:, 0], c.pixels[:, 1]] = True
    return mask


def apply_mask(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Copy of ``frame`` with masked pixels set to ``MASK_COLOR``."""
    out = frame.astype(np.int16, copy=True)
    if mask is not None:
        out[mask] = MASK_COLOR
    return out


def state_key(frame: np.ndarray, mask: np.ndarray | None = None) -> str:
    """Stable 128-bit hex key of the frame with status-bar pixels zeroed."""
    f = frame.astype(np.uint8, copy=True)
    if mask is not None:
        f[mask] = 0
    return hashlib.blake2b(f.tobytes(), digest_size=16).hexdigest()


def click_group(c: Component) -> int:
    """Priority tier 0 (most button-like) .. 4 (status bar) for a component."""
    if c.color == MASK_COLOR:
        return 4
    salient = SALIENT_MIN_COLOR <= c.color < PALETTE_SIZE
    medium = MEDIUM_MIN <= c.width <= MEDIUM_MAX and MEDIUM_MIN <= c.height <= MEDIUM_MAX
    if salient and medium:
        return 0
    if medium:
        return 1
    if salient:
        return 2
    return 3


def click_groups(components: list[Component]) -> list[int]:
    return [click_group(c) for c in components]
