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
| 5 | **Instance-level click rules**: outcomes per object instance (signature + position), not per kind | done — danger rules kept, no-op rules dropped |
| 6 | **Curiosity-directed frontier**: walk towards frontier nodes whose untested actions the model cannot predict | done — no gain, kept as `model-cur` |
| 7 | **Goal induction by lift** over the base rate (colours ahead, click signatures); default agent validated at 4,000 and 12,000 steps | done |
| 8a | **Object-targeted exploration**: when nothing unknown is testable locally, walk to the nearest object the avatar has never touched | done — faster first levels, no extra levels; kept as `model-explore` |
| 8b | **Object-level goals**: the *signature* of the object the avatar entered when a level ended, planned to by signature | done — +1 median level (`dc22`), new default |
| 8c | **Adjacency preconditions**: the objects touching the avatar when a level ended (e.g. the colour-2 blocks of `tu93`), used as plan targets and candidate priorities | proposed next step |
| 9 | (optional) a local language model that proposes hypotheses | — |

**Where this stands.** The default `model` agent beats the exploration baseline at every
budget tried, and the gap widens with the budget (levels per run, all 25 public games):

| step budget | graph exploration | model (this repo) | gain |
|---|---|---|---|
| 4,000 (5 seeds) | 0.60 | 0.74 | +24 % |
| 12,000 (3 seeds) | 0.69 | 0.89 | +29 % |
| 24,000 (2 seeds) | 0.78 | 1.18 | **+51 %** |

At 24,000 steps the sum of per-game median levels is 29.5 against 19.5; the model is
ahead on `lp85` (5 vs 2), `dc22` (3 vs 1), `tu93` (3 vs 1), `vc33` (3.5 vs 2), `ka59`,
`ls20`, `r11l`, and behind only on `lf52` (1.5 vs 2). Full tables: `results/summary.md`.

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

## Results — Phase 6 (curiosity-directed frontier): no gain

With nothing left to test at the current node, the agent first walks towards frontier
nodes that still hold candidates of unknown outcome, and only then towards the nearest
frontier of any kind. It took that detour ~215 times per run and scored 19 / 90 against
18 / 92 for the same agent without it (`model-cur` vs `model-nocur`), losing on `tu93`
(10 vs 12 levels over five seeds). Off by default.

**Where this leaves the agent.** Phases 4–6 each moved the total by a few levels in either
direction, which is the run-to-run noise of this benchmark at 4,000 steps; the gain over
the exploration baseline comes from Phase 2 (rules that prune and order candidates) and
has stayed at about +3 median levels / +20 levels over all runs ever since. The
level-completing transitions (`neo4 explain`, and the diagnostic in the Phase 3 section)
are relational — which instance, where, in what state — so the next phase induces
*preconditions* over object relations instead of colours and signatures.

## Results — Phase 7 (goal induction by lift) and the 12,000-step comparison

A fixed "30 % of visits" threshold never recognised a real goal in these games: the click
that completes a `vc33` level is 1 of ~88 clicks on the rare colour-9 4x4 block, and the
`ls20` exit colour had been bumped into three times before. A feature is now a goal when
its share of level-completing visits is at least **5x the base rate** of all transitions.
`neo4 explain` then reports, for example, `clicking colour-9 4x4 completed a level (2x)`
on `vc33`, `clicking colour-8 6x8` on `lp85`, and `entering colour 11` on `dc22`. A goal
by clicked *colour* was tried and dropped: on `r11l` it spent ~500 steps per run clicking
every colour-15 pixel, where the position decides. The validated default agent
(`model-lift`: rules, instance-level danger, lift goals, no curiosity, no abstraction)
scores **20 / 93** at 4,000 steps — the same as the Phase 3 agent.

**At 12,000 steps the gap over the baseline widens** (3 seeds, median levels per game and
levels over all runs):

| agent | sum of per-game median levels | levels over all runs |
|---|---|---|
| graph exploration | 16 | 52 |
| model (this repo) | **23** | **67** |

Per game the model is ahead on `dc22` (2 vs 1), `ft09` (2 vs 1), `lp85` (2 vs 1), `r11l`
(2 vs 1) and `tu93` (3 vs 0), behind nowhere, and reaches the first level sooner almost
everywhere (`ls20` 198 vs 640 steps, `lp85` 36 vs 156, `lf52` 202 vs 597, `dc22` 805 vs
1,551). The rules compound with the budget: the more steps, the more of them the pruning
saves and the more levels the saved steps buy.

## Results — Phase 8a (object-targeted exploration): faster, not further

When nothing of unknown outcome is left to test at the current node, the agent plans a
route to the nearest object the avatar has never touched and goes there, instead of
grinding through moves whose outcome it can predict. On a toy open room this reaches the
goal in ~55 steps against ~650 for exhaustive exploration, and on `ls20` the first level
falls in 200 steps instead of 604. It fires on few public games, though (`ls20`, `re86`,
`bp35`, `tu93`), buys no additional level anywhere at 4,000 steps (19 / 90 against
20 / 93), and costs on `tu93` (10 vs 13 levels over five seeds), where "touching" an
object pushes it and scrambles the puzzle. Off by default; `model-explore` keeps it.

Two other things the diagnostics settled: `tn36` and `cn04` always lose at exactly 61 and
75 steps — a step limit, nothing to avoid — and `sp80` loses on ACTION5 145 times per
run but on only 20 % of its uses, with nothing in the avatar's surroundings separating
the fatal uses from the safe ones; an action-level danger rule (an action that loses the
game on half its uses) is in place but does not fire there.

## Results — Phase 8b (object-level goals): +1 median level

A level-completing move now records the *signature* of the object the avatar entered,
not just its colour, with lift measured against how often that signature was ahead of
the avatar. `neo4 explain` reports e.g. `entering the colour-5 7x7 sprite completed a
level (1/2 contacts)` on `ls20` and `entering the colour-11 2x2 sprite` on `dc22`, and
lists the objects touching the avatar at that moment (`tu93`: colour-2 3x3 blocks, three
times out of three). The planner targets goal objects by signature and falls back to
colours; in a test room whose walls share the exit's colour, the colour goal plans into
the walls (every plan fails) while the signature goal reaches the exit. A second fix
mattered as much: a failed route search no longer spends the per-level planning
allowance, which it used to exhaust before the agent had learned the new level's floor.

Result at 4,000 steps, 5 seeds: **21 / 97** against 20 / 93 for the previous default, the
whole difference on `dc22` (median 1 → 2, 5 → 9 levels over seeds, one plan per run and
no failed plan); no game is worse. This is the new default agent.

## Conclusion and next step

What moved the needle was Phase 2: rules induced from the agent's own transitions —
which object the keyboard moves and how far, which colours block it, which clicks never
do anything, which actions or cells lose the game — used to prune and order the
exploration. Everything after that (planning towards past goals, state abstraction,
curiosity, instance-level rules, object-targeted exploration) was measured and found
neutral or negative at 4,000 steps, and is kept only as an opt-in variant. The gain from
the rules grows with the budget: +24 % levels per run at 4,000 steps, +51 % at 24,000.

The remaining losses are of one kind. What completes a level, and what loses the game,
is relational — *which* instance of an object, *where*, in *what* state, *pushed* against
what — and a model over colours and signatures cannot express it. Phase 8b took the first
step (goals as object signatures, +1 median level); the proposed next one (Phase 8c) uses
the objects *touching* the avatar at level-completing moves, which `neo4 explain` already
reports, as plan targets and candidate priorities — `tu93` is the test case.

The step budget of the ARC Prize 2026 Kaggle track is not public at the time of writing;
the agent runs at about 60–250 engine steps per second on one CPU core depending on the
game, with no GPU and no network.

## Attribution and licence

Phase 1 is an independent reimplementation of *Graph-Based Exploration for ARC-AGI-3
Interactive Reasoning Tasks* (AAAI 2026 workshop) and its open-source code
[`arc-agi-3-just-explore`](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore) (MIT).
Game environments are provided by the ARC Prize Foundation through the `arc-agi` toolkit.

This repository is public and MIT-licensed (see `LICENSE`). ARC Prize requires every
eligible entry to be open source, and nothing here is secret: API keys, downloaded game
files and raw run logs are git-ignored.
