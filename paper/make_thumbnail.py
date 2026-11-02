"""The writeup's card and thumbnail image, 560 x 280 as Kaggle asks. The card shows all of it; the
thumbnail is a square cut from it, so the left half is a square that stands alone (the title) and
the right half carries the scores. Writes paper/thumbnail.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Inter"
SURFACE, INK, INK2, SERIES, ON_SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#ffffff"
STATS = [  # value, label (two lines)
    ("80.01", "public games, each\nwith its own solver"),
    ("0.70", "the same games,\nwithout their solvers"),
    ("0.46", "hidden games\n(Kaggle leaderboard)"),
]

fig = plt.figure(figsize=(5.6, 2.8), dpi=100, facecolor=SURFACE)
fig.patches.append(Rectangle((0, 0), 0.5, 1, transform=fig.transFigure, facecolor=SERIES, linewidth=0))
fig.text(0.045, 0.89, "Search\nis not the\nbottleneck", fontsize=27, fontweight="bold", color=ON_SERIES,
         va="top", linespacing=1.05)
fig.text(0.045, 0.09, "A frames-only ARC-AGI-3 agent,\nno language model", fontsize=9, color=ON_SERIES,
         alpha=0.9, linespacing=1.3)
for i, (value, label) in enumerate(STATS):
    y = 0.80 - i * 0.235
    fig.text(0.545, y, value, fontsize=19, fontweight="bold", color=INK, va="center")
    fig.text(0.735, y, label, fontsize=8, color=INK2, va="center", linespacing=1.25)
fig.text(0.545, 0.09, "A level learned in L lives is worth\nabout 1/L² of its share", fontsize=8.5, color=INK,
         linespacing=1.3)
out = Path(__file__).resolve().parent / "thumbnail.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
