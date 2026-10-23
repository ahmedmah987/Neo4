"""The public games grouped into mechanic families, with each family's numbers.

    python scripts/families.py [results/raw/model.jsonl]

The grouping was made by reading the public games' sources and first frames
(README, "The mechanics library"). It is documentation of the public set, not
an input to the agent: the solvers in neo4/library.py recognise their family
from the frames alone, because the evaluation games are different games.

For each family the script prints its games, levels, the summed human
baselines, what the agent spends on the levels it completes, its current
official-style contribution to the 25-game mean, and the most the family can
contribute (100 points per game at human cost, over 25 games).
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neo4.run import RAW, _baselines, official_score  # noqa: E402

FAMILIES: dict[str, tuple[str, list[str]]] = {
    "navigate": ("Navigate an avatar to an exit (keys, switches, hazards, budgets)",
                 ["ls20", "tu93", "sc25", "g50t", "dc22", "bp35", "m0r0"]),
    "sokoban": ("Push or carry blocks into marked zones",
                ["wa30", "ka59"]),
    "marker": ("Bring pieces onto markers by indirect control (no avatar)",
               ["lp85", "re86", "s5i5", "r11l", "vc33", "ar25", "su15"]),
    "copy": ("Reproduce a shown pattern or sequence",
             ["cd82", "tr87", "sk48"]),
    "constraint": ("Satisfy local rules on a board (toggles, pegs, jigsaw)",
                   ["ft09", "lf52", "cn04"]),
    "program": ("Compose a program or route a flow, then run it",
                ["tn36", "sb26", "sp80"]),
}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW / "model.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    base = _baselines()
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["game"], []).append(r)
    n_games = sum(len(g) for _, g in FAMILIES.values())
    print("| Family | Games | Levels | Human actions (all levels) | Levels the agent completes "
          "(mean of seeds) | Agent actions on those levels / human | Contribution now | Most it can add |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    total_now = 0.0
    for key, (desc, games) in FAMILIES.items():
        levels = sum(len(base[g]) for g in games)
        human = sum(sum(base[g]) for g in games)
        done = sum(st.mean(len(r["level_steps"]) for r in by[g]) for g in games if by.get(g))
        ours = theirs = 0.0
        for g in games:
            for r in by.get(g, []):
                prev = 0
                for i, s in enumerate(r["level_steps"]):
                    ours += s - prev
                    theirs += base[g][i]
                    prev = s
        ratio = f"{ours / theirs:.1f}x" if theirs else "-"
        now = sum(st.mean(official_score(r, base[g]) for r in by[g]) for g in games if by.get(g)) / n_games
        total_now += now
        print(f"| {desc} | {', '.join(games)} | {levels} | {human:,} | {done:.1f} | {ratio} | "
              f"{now:.2f} | {100 * len(games) / n_games:.0f} |")
    print(f"\nmean official-style score now: {total_now:.2f}")


if __name__ == "__main__":
    main()
