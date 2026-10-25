"""Replay a diagnosed game's solutions (results/diagnosis/<game>.json) from the reset to the win,
chained in one environment, and print each level's actions against the humans' and the game's
official-style score. Usage: python scripts/replay_diagnosis.py <game> [<game> ...]

Actions are U D L R S Z (keys 1, 2, 3, 4, 5 and 7) and "C x y" (a click at a frame pixel)."""

import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.INFO)

from arcengine import GameAction  # noqa: E402

from neo4.run import arcade  # noqa: E402

KEYS = {"U": GameAction.ACTION1, "D": GameAction.ACTION2, "L": GameAction.ACTION3,
        "R": GameAction.ACTION4, "S": GameAction.ACTION5, "Z": GameAction.ACTION7}


def replay(game: str) -> float:
    sol = json.loads((ROOT / "results" / "diagnosis" / f"{game}.json").read_text())["levels"]
    env = arcade().make(game, seed=0)
    obs = env.reset()
    total, weighted, weights = 0, 0.0, 0
    for lv in sorted(sol, key=int):
        level = sol[lv]
        n = 0
        for tok in re.findall(r"C \d+ \d+|[UDLRSZ]", level["actions"]):
            if tok.startswith("C"):
                _, x, y = tok.split()
                obs = env.step(GameAction.ACTION6, {"x": int(x), "y": int(y)})
            else:
                obs = env.step(KEYS[tok])
            n += 1
            if obs.levels_completed >= int(lv) or obs.state.name != "NOT_FINISHED":
                break
        done = obs.levels_completed >= int(lv)
        score = min(115.0, (level["human"] / n) ** 2 * 100) if done else 0.0
        weighted += int(lv) * score
        weights += int(lv)
        total += n
        print(f"{game} level {lv}: {n} actions (humans {level['human']}), "
              f"{'completed' if done else 'NOT completed'}, {obs.state.name}")
        if not done:
            break
    print(f"{game}: {total} actions, game score {weighted / weights:.2f}")
    return weighted / weights


if __name__ == "__main__":
    for g in sys.argv[1:]:
        replay(g)
