"""The writeup's cover image: the official-style score of the agent where its solvers were built
for the game, and where they were not. Writes paper/cover.png (1920 x 1080). Kaggle's Media
Gallery shows images whole in a 16:9 frame about 900 pixels wide, so the image is 16:9 (no bars
above and below) and its smallest text is about 20 pixels in 1,600."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Inter"
SURFACE, INK, INK2, MUTED, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#2a78d6"
ROWS = [  # label, official-style score
    ("The 25 public games, each with the solver built for it", 80.01),
    ("The same games, each without its own solver (holdout)", 0.70),
    ("The model-guided explorer alone", 0.44),
    ("The hidden games (Kaggle leaderboard, final submissions v7 and v10)", 0.46),
]

fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURFACE)
fig.text(0.06, 0.865, "Search is not the bottleneck", fontsize=48, fontweight="bold", color=INK)
fig.text(0.06, 0.745, "A mechanics library for ARC-AGI-3 plays near human efficiency on the games it was\n"
         "built for, and almost not at all on the others.", fontsize=21, color=INK2, linespacing=1.35)
ax = fig.add_axes([0.06, 0.205, 0.88, 0.50], facecolor=SURFACE)
n = len(ROWS)
for i, (label, value) in enumerate(ROWS):
    y = n - 1 - i
    ax.text(0.8, y + 0.30, label, fontsize=20, color=INK, va="bottom")
    width = max(value, 0.15)  # a sliver, so the smallest still shows where it starts
    ax.add_patch(FancyBboxPatch((0, y - 0.15), width, 0.32, boxstyle="round,pad=0,rounding_size=0.12",
                                mutation_aspect=1 / 25, linewidth=0, facecolor=SERIES))
    ax.text(width + 1.2, y + 0.01, f"{value:.2f}", fontsize=24, fontweight="bold", color=INK, va="center")
ax.set_xlim(0, 112)
ax.set_ylim(-0.5, n - 0.1)
ax.axis("off")
ax.plot([0, 0], [-0.4, n - 0.35], color=MUTED, linewidth=1)
fig.text(0.06, 0.105, "Why: a level learned in L lives is worth about 1/L² of its share (Spearman 0.83 over 123 "
         "holdout levels);\n92% of the actions go to lives that die learning the way.", fontsize=16, color=INK,
         linespacing=1.35)
fig.text(0.06, 0.045, "Score: per level min(115, (human actions / agent actions)² × 100), weighted by level, "
         "averaged over games · github.com/ahmedmah987/neo4", fontsize=13.5, color=MUTED)
out = Path(__file__).resolve().parent / "cover.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
