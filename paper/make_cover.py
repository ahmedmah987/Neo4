"""The writeup's cover image: the official-style score of the agent where its solvers were built
for the game, and where they were not. Writes paper/cover.png (1600 x 800)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, INK2, MUTED, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#2a78d6"
ROWS = [  # label, official-style score
    ("The 25 public games, each with the solver built for it", 80.01),
    ("The same games, each without its own solver (holdout)", 0.45),
    ("The model-guided explorer alone", 0.44),
    ("The hidden games (Kaggle leaderboard)", 0.33),
]

fig = plt.figure(figsize=(16, 8), dpi=100, facecolor=SURFACE)
fig.text(0.05, 0.88, "Search is not the bottleneck", fontsize=40, fontweight="bold", color=INK)
fig.text(0.05, 0.81, "A mechanics library for ARC-AGI-3 plays near human efficiency on the games it was built "
         "for, and almost not at all on the others.", fontsize=17, color=INK2)
ax = fig.add_axes([0.05, 0.17, 0.90, 0.56], facecolor=SURFACE)
n = len(ROWS)
for i, (label, value) in enumerate(ROWS):
    y = n - 1 - i
    ax.text(0.8, y + 0.33, label, fontsize=17, color=INK, va="bottom")
    width = max(value, 0.15)  # a sliver, so the smallest still shows where it starts
    ax.add_patch(FancyBboxPatch((0, y - 0.13), width, 0.30, boxstyle="round,pad=0,rounding_size=0.12",
                                mutation_aspect=1 / 25, linewidth=0, facecolor=SERIES))
    ax.text(width + 1.2, y + 0.02, f"{value:.2f}", fontsize=20, fontweight="bold", color=INK, va="center")
ax.set_xlim(0, 118)
ax.set_ylim(-0.5, n - 0.1)
ax.axis("off")
ax.plot([0, 0], [-0.4, n - 0.35], color=MUTED, linewidth=1)
fig.text(0.05, 0.08, "Official-style score: per level min(115, (human actions / agent actions)² × 100), levels weighted "
         "by their index, averaged over games.", fontsize=13, color=INK2)
fig.text(0.05, 0.045, "Neo4 · a frames-only agent, no language model · MIT-0 · github.com/ahmedmah987/neo4",
         fontsize=13, color=MUTED)
out = Path(__file__).resolve().parent / "cover.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
