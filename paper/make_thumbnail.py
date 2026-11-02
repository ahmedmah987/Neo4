"""The writeup's card and thumbnail image, 560 x 280 as Kaggle asks: the cover's message with
only what still reads at that size. Writes paper/thumbnail.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Inter"
SURFACE, INK, INK2, MUTED, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#2a78d6"
STATS = [  # value, label (two lines)
    ("80.01", "public games,\neach with its own solver"),
    ("0.70", "the same games,\nwithout their solvers"),
    ("0.46", "hidden games\n(Kaggle leaderboard)"),
]

fig = plt.figure(figsize=(5.6, 2.8), dpi=100, facecolor=SURFACE)
fig.text(0.06, 0.80, "Search is not the bottleneck", fontsize=20, fontweight="bold", color=INK)
fig.text(0.06, 0.69, "A frames-only ARC-AGI-3 agent, no language model", fontsize=10.5, color=INK2)
for i, (value, label) in enumerate(STATS):
    x = 0.06 + i * 0.31
    fig.text(x, 0.41, value, fontsize=25, fontweight="bold", color=SERIES if i == 0 else INK)
    fig.text(x, 0.36, label, fontsize=8.5, color=INK2, va="top", linespacing=1.25)
fig.text(0.06, 0.07, "A level learned in L lives is worth about 1/L² of its share", fontsize=9.5, color=INK)
out = Path(__file__).resolve().parent / "thumbnail.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
