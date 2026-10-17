# Neo4 — an agent for ARC-AGI-3

An exploration + rule-induction agent for the [ARC-AGI-3](https://arcprize.org/arc-agi/3/)
interactive benchmark.

ARC-AGI-3 is a set of small, game-like environments with **no instructions**: the agent
has to discover what the goal is and how the world works by interacting with it. Humans
solve all of them; frontier language models used directly solve almost none.

## Plan

| Phase | Goal | Status |
|---|---|---|
| 1 | Reimplement the training-free **graph-exploration** baseline (Rudakov, Shock & Cowley, [arXiv:2512.24156](https://arxiv.org/abs/2512.24156)) and measure it on the public games | done |
| 2 | **Rule induction**: learn what each action does to each object on screen from the recorded transitions | done |
| 3 | **Planning** inside the induced model; measure the gain over the exploration baseline | done — no gain, see below |
| 4 | **State abstraction**: learn which objects change independently of the agent's actions and leave them out of the state identity | done — net loss, kept as the `model-abs` ablation |
| 5 | **Instance-level click rules**: outcomes per object instance (signature + position), not per kind | in progress |
| 6 | **Curiosity-directed frontier**: walk towards frontier nodes whose untested actions the model cannot predict | next |
| 7 | (optional) a local language model that proposes hypotheses | — |

## Setup and usage

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                                                   # unit tests
neo4 run --agent graph --games ls20 vc33 --steps 4000 --seeds 3
neo4 run --agent random --games all --steps 4000 --seeds 5
neo4 summarize                                           # writes results/summary.md
```

No GPU and no API key are needed: the official `arc-agi` toolkit downloads the public
games with an anonymous key on first use and runs them locally (about 1,500 engine
steps per second on a laptop-class CPU). After the first download the runner works offline.

## Layout

```
neo4/
  frame.py        frame processing: connected components, status-bar masking,
                  state keys, click candidates in five priority tiers
  graph_agent.py  graph-exploration agent (Algorithm 1 of the paper) and a random control agent
  objects.py      objects with signatures; matching two frames into moved / vanished / appeared
  dynamics.py     rule induction from transitions; predictions per candidate; English report
  planner.py      BFS path planning for the avatar through cells the model knows to be passable
  model_agent.py  the graph agent guided by the induced rules ("model"), plus a no-planning ablation
  run.py          runner: games x seeds under a step budget, multi-process; summariser; `explain`
results/          summarised result tables (raw per-run JSON lines are git-ignored)
tests/            unit tests for every module, on synthetic frames and toy environments
```

`neo4 explain --game ls20` plays a game with the model agent and prints what it learned, e.g.

```
avatar: colour-9 3x5 sprite
  ACTION1: moves it by (0, -5) (745/748 times; stuck 3x)
  ACTION3: moves it by (-5, 0) (477/479 times; stuck 2x)
  colour 3 is walkable (1889/1903)
  colour 5 blocks the avatar (3/3)
```

## How the exploration agent works

* Every distinct frame (after masking probable status bars) is a node in a per-level graph.
* Every candidate action at a node is an edge: the available keyboard actions, plus one
  click per connected component for click games. Candidates carry a priority group
  (0 = most button-like, 4 = probable status bar).
* Algorithm 1: run an untested candidate at the current node if one exists within the
  current priority threshold; otherwise take the first step of the shortest known path to
  a node that has one; otherwise raise the threshold.
* Actions that end the game or return the level to its first frame are still recorded
  as tested, which avoids the loop reported in the paper's official run.
* If the whole reachable graph is tested without a level-up (the state key misses hidden
  state), the graph is *reopened*: a new epoch re-tests everything in a fresh random
  order while keeping the known transitions for path finding.

## How the model-guided agent works

The model agent is the exploration agent plus rules it induces from its own transitions.

* **Objects.** A frame is a set of objects: single-colour connected components with a
  signature (colour, size, and the exact shape for small sprites). Two consecutive frames
  are matched into *moved*, *vanished*, *appeared* and *recoloured* objects; large regions
  are static scenery.
* **Rules** (`dynamics.py`), kept for the whole game so they transfer between levels:
  the *avatar* (the signature the keyboard moves most), the displacement each action gives
  it, which colours block it or let it pass, which colours ahead of it ended the game,
  and which click targets never change anything.
* **Pruning.** Before testing a node's candidates, each one gets a verdict. A candidate
  predicted to do nothing (a known wall ahead, a click target that never did anything) or
  to lose the game is counted as tested without being executed — in the first epoch only,
  so a wrong rule costs at most one re-test when the graph is reopened. Candidates with an
  unknown outcome are tested before predictable ones.
* **Goal transfer and planning** (`planner.py`). When a level is completed, the model
  records what did it: the colour the avatar entered, or the signature clicked. In later
  levels the agent plans a path to that colour through cells it knows to be passable and
  follows it while every step lands where the model predicted; a surprise abandons the
  plan and hands control back to exploration. Clicks on a level-completing signature are
  tested first.

## Results — Phase 1

Setting: all 25 public games, 5 seeds each, **4,000 engine steps per run** (every `step`
call counts, resets included), median over seeds. One run takes 2–15 s; the whole sweep
takes about 11 minutes on 4 CPU cores. Full per-game table: [`results/summary.md`](results/summary.md).

| agent | sum of per-game median levels | games with ≥ 1 level |
|---|---|---|
| graph exploration (this repo) | **17** | 14 / 25 |
| random actions | 6 | 6 / 25 |

Per-game median levels, graph / random (games where both scored 0 omitted):

| game | levels | graph | random |
|---|---|---|---|
| vc33 | 7 | 2 | 0 |
| tu93 | 9 | 2 | 0 |
| cd82 | 6 | 2 | 0 |
| ar25, dc22, lf52, ls20, s5i5, su15 | 8, 6, 10, 7, 8, 9 | 1 | 0 |
| ft09, lp85, m0r0, r11l, sp80 | 6, 8, 6, 6, 6 | 1 | 1 |
| cn04 | 6 | 0 | 1 |

**Caveats.** These numbers are not comparable with Table 1 of the paper (e.g. vc33 = 5,
ls20 = 2 at 4,000 steps): those were measured on the 2025 preview versions of six games,
while the current public set has 25 games with new versions and levels. Several design
details of the baseline are not specified in the paper and were chosen here (deterministic
click point nearest the component centre, no retry of no-op actions, epoch-based reopening).

**What the diagnostics say.** Three failure modes dominate and motivate Phase 2:

1. *State explosion.* `cn04`, `re86` and `tr87` reach 2,000–2,700 distinct states within
   4,000 steps without finishing their first level. Exhaustive exploration cannot scale;
   the agent has to learn what actions *do* instead of trying them everywhere.
2. *Hidden state.* `g50t` and `ls20` produce inconsistent transitions (same frame, same
   action, different outcome) and hundreds of random walks: the masked frame is not the
   full state.
3. *Game-over heavy levels.* `sp80`, `r11l`, `ft09` and `tn36` lose the game 60–90 times
   per run, and every loss costs a path back to the frontier.

## Results — Phase 2 (rule induction, no planning)

Same setting: 25 games, 5 seeds, 4,000 steps. `model-noplan` is the exploration agent with
the induced rules used only to prune and order candidates.

| agent | sum of per-game median levels | games with ≥ 1 level |
|---|---|---|
| model-noplan (rules) | **20** | 15 / 25 |
| graph exploration | 17 | 14 / 25 |
| random actions | 6 | 6 / 25 |

Where it differs: `lp85` 1 → 2, `r11l` 1 → 2, `tu93` 2 → 3, and the first level of `ls20`,
`m0r0` and `su15` is now completed in every seed instead of most. No game got worse.
The rules also make the first level cheaper almost everywhere (median steps to complete
it, graph → model-noplan): `lp85` 129 → 36, `lf52` 670 → 175, `cd82` 1,070 → 474,
`dc22` 1,551 → 810, `s5i5` 784 → 556, `sp80` 86 → 63.

What the induced rules reveal about the hard games (from `neo4 explain`): `tr87`'s state
explosion is caused by colour-7 single pixels that drift one cell per step — moving
particles make every frame unique; `cn04` is a pushing game in which the colour-8 avatar
moves colour-10 bars and is stuck 60 % of the time because whether a push works depends on
what lies beyond the bar, a conditional rule the current colour-level model cannot express.

## Results — Phase 3 (goal transfer and planning): no gain

The full `model` agent (pruning + goal transfer + path planning) scores exactly the same
as `model-noplan`: 20 median levels, 94 levels over all runs, identical per-game numbers.
The planner ran on one game only (`ls20`: 5 plans per run, none failed, none completed a
level) and the "click the signature that completed a level before" rule never fired.

The reason is visible in the statistics. A level-completing event is recorded as a *colour*
the avatar entered or a *signature* that was clicked, and in these games that is not what
completes a level:

* in `ls20` the colour entered on level completion is the ordinary floor; the exit works
  only once a key has been collected, so re-entering that colour does nothing;
* in `vc33` the level-completing click lands on a colour-9 2x2 block, the same signature
  as the ~1,900 other clicks in the run, so the signature carries no information about
  *which* block or *when*.

What completes a level is relational — which object, in what state, relative to what —
and a colour- or signature-level goal cannot express it. The planner itself works (it is
exercised on a toy room in the tests, where the second level is solved in a quarter of
the steps of the first), but on the public games it has nothing useful to plan towards
yet. Goal induction needs a richer representation; that is left for later, and the next
phase targets the larger measured loss, the state explosion.

## Results — Phase 4 (state abstraction): net loss, now opt-in

Idea: objects that move, appear or vanish whatever the agent does (drifting particles, an
in-scene timer, an animation) multiply the number of states without carrying information,
so learn which signatures are such *noise* and paint them out before hashing the frame.

| agent | sum of per-game median levels | levels over all runs |
|---|---|---|
| model-plan (Phase 3, no abstraction) | 20 | 94 |
| model-abs-lax: noise = churns on ≥ 30 % of steps under every action class | 13 | 65 |
| model-abs: clicks on different targets are different classes, ≥ 50 %, ≥ 2 classes | 16 | 83 |

The lax criterion treated every click as one action, so in click games any object that
reacts to clicks on many targets — the game's actual mechanics — was erased, and whole
levels collapsed into a single state (`lp85`: one node, 0 levels). The strict criterion
repairs most of that but still loses `tu93` (3 → 0) and `r11l` (2 → 1), and where it does
collapse a state space (`re86`: 2,590 → 25 nodes; `ka59`: 1,039 → 78) no level follows:
the "noise" there is an animation whose phase the game cares about. Abstraction stays
available as `model-abs` but is off in the default agent. Two further checks: unmasking the
status bar in `ls20` removes the inconsistent transitions but explodes the state space
(140 → 2,400 nodes), because the bar is a step counter — masking it is the right trade;
`g50t` is inconsistent with or without the mask, i.e. it has genuine hidden state.

## Results — Phase 5 (instance-level click rules): mixed, kept in part

Lost games in the click games are spread over the *instances* of one signature (`r11l`:
78 of 704 clicks on colour-15 pixels lose, i.e. some cells kill and most do not), so
signature-level rules never fire. Keeping click outcomes per object instance (signature +
position) as well, and skipping instance-level no-ops and dangers, gave 18 median levels /
91 over all runs (`model-inst`) against 20 / 94: fewer lost games on `r11l` (112 → 79) but
fewer levels there too (2 → 1), and `su15` dropped (5 → 3 over all runs), while `ft09`
(4 → 6), `lf52` and `ka59` gained. The instance-level *no-op* rule is the culprit: a cell
that did nothing three times may be exactly the cell that matters once the state has
changed. The default agent now keeps instance-level *danger* rules only.

## Attribution

Phase 1 is an independent reimplementation of *Graph-Based Exploration for ARC-AGI-3
Interactive Reasoning Tasks* (AAAI 2026 workshop) and its open-source code
[`arc-agi-3-just-explore`](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore) (MIT).
Game environments are provided by the ARC Prize Foundation through the `arc-agi` toolkit.
