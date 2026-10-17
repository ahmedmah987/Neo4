"""Run agents on ARC-AGI-3 games and summarise the results.

    neo4 run --agent graph --games all --steps 4000 --seeds 5
    neo4 summarize

``run`` appends one JSON line per (game, seed) to ``results/raw/<agent>.jsonl``;
``summarize`` turns every raw file into ``results/phase1.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from arcengine import GameState

from .graph_agent import AGENTS as _GRAPH_AGENTS
from .model_agent import (
    ModelAbstractAgent,
    ModelAdjacencyAgent,
    ModelAgent,
    ModelCuriousAgent,
    ModelExploreAgent,
    ModelNoPlanAgent,
)

AGENTS = {
    **_GRAPH_AGENTS,
    ModelAgent.name: ModelAgent,
    ModelNoPlanAgent.name: ModelNoPlanAgent,
    ModelAbstractAgent.name: ModelAbstractAgent,
    ModelCuriousAgent.name: ModelCuriousAgent,
    ModelExploreAgent.name: ModelExploreAgent,
    ModelAdjacencyAgent.name: ModelAdjacencyAgent,
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
ENV_DIR = ROOT / "environment_files"

PUBLIC_GAMES = [
    "ls20", "ar25", "lp85", "m0r0", "tn36", "bp35", "sb26", "tu93", "vc33",
    "cn04", "r11l", "ft09", "ka59", "cd82", "sc25", "wa30", "g50t", "su15",
    "tr87", "dc22", "s5i5", "sp80", "sk48", "lf52", "re86",
]


_arcade: Any = None


def _quiet_logger() -> logging.Logger:
    """The toolkit installs its own INFO handler on stdout; hand it a quiet one instead."""
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    log = logging.getLogger("neo4.arcade")
    log.setLevel(logging.WARNING)
    return log


def arcade() -> Any:
    """One Arcade per process. Offline once the game files are on disk."""
    global _arcade
    if _arcade is None:
        import arc_agi

        mode = (
            arc_agi.OperationMode.OFFLINE
            if any(ENV_DIR.rglob("metadata.json"))
            else arc_agi.OperationMode.NORMAL
        )
        _arcade = arc_agi.Arcade(
            environments_dir=str(ENV_DIR), operation_mode=mode, logger=_quiet_logger()
        )
    return _arcade


def download_games(games: list[str]) -> None:
    """Fetch any game not yet on disk (needs network, anonymous key is fine)."""
    import arc_agi

    have = {p.parent.parent.name for p in ENV_DIR.rglob("metadata.json")}
    missing = [g for g in games if g not in have]
    if not missing:
        return
    arc = arc_agi.Arcade(environments_dir=str(ENV_DIR), logger=_quiet_logger())
    for g in missing:
        if arc.make(g) is None:
            raise RuntimeError(f"could not download game {g}")


def play(
    agent_name: str, game: str, seed: int, steps: int, agent: Any = None, label: str | None = None
) -> dict[str, Any]:
    """Play one game with a fresh agent under a step budget."""
    env = arcade().make(game, seed=seed)
    if env is None:
        raise RuntimeError(f"could not create environment for {game}")
    if agent is None:
        agent = AGENTS[agent_name](seed=seed)

    obs = env.reset()
    assert obs is not None
    t0 = time.time()
    level_steps: list[int] = []
    best = 0
    used = 0
    for used in range(1, steps + 1):
        action, data = agent.act(obs)
        obs = env.step(action, data)
        assert obs is not None
        if obs.levels_completed > best:
            level_steps.extend([used] * (obs.levels_completed - best))
            best = obs.levels_completed
        if obs.state == GameState.WIN:
            break
    return {
        "agent": label or agent_name,
        "game": game,
        "seed": seed,
        "budget": steps,
        "steps_used": used,
        "levels": best,
        "win_levels": obs.win_levels,
        "won": obs.state == GameState.WIN,
        "level_steps": level_steps,
        "seconds": round(time.time() - t0, 2),
        "stats": dict(getattr(agent, "stats", {})),
    }


def _job(args: tuple[str, str, int, int, str]) -> dict[str, Any]:
    agent, game, seed, steps, label = args
    try:
        return play(agent, game, seed, steps, label=label)
    except Exception as e:  # keep the pool alive; report the failure as a row
        return {"agent": label, "game": game, "seed": seed, "budget": steps, "error": repr(e)}


def cmd_run(a: argparse.Namespace) -> None:
    games = PUBLIC_GAMES if a.games == ["all"] else a.games
    download_games(games)
    label = a.label or a.agent
    jobs = [(a.agent, g, s, a.steps, label) for g in games for s in range(a.seeds)]
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{label}.jsonl"
    workers = a.workers or os.cpu_count() or 1
    print(f"{len(jobs)} runs, {workers} workers -> {out}", file=sys.stderr)
    with Pool(workers) as pool, out.open("a") as f:
        for row in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if "error" in row:
                print(f"{row['game']} seed={row['seed']}: ERROR {row['error']}", file=sys.stderr)
            else:
                print(
                    f"{row['game']} seed={row['seed']}: {row['levels']}/{row['win_levels']} "
                    f"levels in {row['steps_used']} steps ({row['seconds']}s)",
                    file=sys.stderr,
                )


def _load_raw() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW.glob("*.jsonl")):
        with path.open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return [r for r in rows if "error" not in r]


def cmd_summarize(_: argparse.Namespace) -> None:
    rows = _load_raw()
    if not rows:
        print("no results under results/raw", file=sys.stderr)
        return
    agents = sorted({r["agent"] for r in rows})
    games = sorted({r["game"] for r in rows})
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["game"], r["agent"]), []).append(r)
    budget = sorted({r["budget"] for r in rows})

    lines = [
        "# Results on the public games",
        "",
        f"Step budget per run: {', '.join(map(str, budget))}. "
        "Cell = median levels completed over seeds (min–max), out of the game's level count.",
        "",
        "| game | levels | " + " | ".join(agents) + " |",
        "|---|---|" + "---|" * len(agents),
    ]
    totals = {ag: [] for ag in agents}
    solved = {ag: 0 for ag in agents}
    for g in games:
        cells = []
        n_levels = next(r["win_levels"] for r in rows if r["game"] == g)
        for ag in agents:
            runs = by.get((g, ag), [])
            if not runs:
                cells.append("—")
                continue
            levels = [r["levels"] for r in runs]
            med = statistics.median(levels)
            totals[ag].append(med)
            solved[ag] += sum(levels)
            cells.append(f"{med:g} ({min(levels)}–{max(levels)}) n={len(runs)}")
        lines.append(f"| {g} | {n_levels} | " + " | ".join(cells) + " |")
    lines.append(
        "| **sum of medians** | | "
        + " | ".join(f"**{sum(totals[ag]):g}**" for ag in agents)
        + " |"
    )
    lines.append(
        "| levels solved over all runs | | " + " | ".join(str(solved[ag]) for ag in agents) + " |"
    )

    # Efficiency: how many steps it took to finish the first level, where it was finished.
    lines += [
        "",
        "Median steps to complete the first level, over the seeds that completed it "
        "(count in brackets); games nobody finished a level in are omitted.",
        "",
        "| game | " + " | ".join(agents) + " |",
        "|---|" + "---|" * len(agents),
    ]
    for g in games:
        cells = []
        any_hit = False
        for ag in agents:
            hits = [r["level_steps"][0] for r in by.get((g, ag), []) if r.get("level_steps")]
            if hits:
                any_hit = True
                cells.append(f"{statistics.median(hits):g} [{len(hits)}]")
            else:
                cells.append("—")
        if any_hit:
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_explain(a: argparse.Namespace) -> None:
    """Play one game with the model agent and print the rules it induced."""
    download_games([a.game])
    agent = ModelAgent(seed=a.seed)
    row = play(ModelAgent.name, a.game, a.seed, a.steps, agent=agent)
    print(f"{a.game}: {row['levels']}/{row['win_levels']} levels in {row['steps_used']} steps; {row['stats']}")
    for line in agent.dyn.describe():
        print(line)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="neo4")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play games with an agent")
    r.add_argument("--agent", choices=sorted(AGENTS), default="graph")
    r.add_argument("--games", nargs="+", default=["all"], help="game ids or 'all'")
    r.add_argument("--steps", type=int, default=4000, help="step budget per run")
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--workers", type=int, default=0, help="processes (default: all cores)")
    r.add_argument("--label", default="", help="row/file label instead of the agent name")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize", help="write results/phase1.md from raw runs")
    s.set_defaults(func=cmd_summarize)
    e = sub.add_parser("explain", help="print the rules the model agent learns on a game")
    e.add_argument("--game", required=True)
    e.add_argument("--steps", type=int, default=2000)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_explain)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
