# Neo4 — an agent for ARC-AGI-3

An exploration + rule-induction agent for the [ARC-AGI-3](https://arcprize.org/arc-agi/3/)
interactive benchmark.

ARC-AGI-3 is a set of small, game-like environments with **no instructions**: the agent
has to discover what the goal is and how the world works by interacting with it. Humans
solve all of them; frontier language models used directly solve almost none.

The project has two strands that share one codebase:

* **"Beyond"** — a working study of what comes after today's language models: an agent
  that learns a *model of its world* from its own interaction (which object it controls,
  what each action does, what blocks, what kills, what completes a level), plans inside
  that model, and keeps what it learned across levels. Every phase below is an experiment
  on that question, with its negative results kept.
* **The competition** — the same agent packaged for the ARC Prize 2026 Kaggle track
  (`kaggle/my_agent.py`), to be submitted once its score is high enough to be worth it.
  The two strands pull the same way: what raises the score is a better world model.

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
| 8c | **Adjacency preconditions**: the objects touching the avatar when a level ended, and the key that completed it, used as plan targets and candidate priorities | done — neutral (20 / 97 vs 21 / 97), kept as `model-adj` |
| 8d | **Object–object relations**: what a pushed object ended up touching or covering when a level ended (`tu93`'s blocks on their targets) | proposed next step |
| 9 | **Competition packaging**: the agent as the single file the Kaggle notebook embeds, tested against the framework contract | done — `kaggle/my_agent.py` |
| 10 | **Robust avatar identification**: orientation-free sprite signatures, the avatar is usually unique; exact signatures kept for click rules | done — neutral in levels (20 / 95), kept: the avatar is now right |
| 11 | **Swept-cell world model**: an avatar that jumps several cells is stopped by what lies *between* its position and its destination; a stopped move blames the swept colours not yet known to be crossable; a graph edge that led somewhere more often than it killed survives a loss (move budgets) | done — 21 / 98, official-style score 0.42 (from 0.39), new default |
| 12 | **Death attribution**: a level's move budget is detected and its losses blame nothing; the frame of a lost game names the kind that moved into the avatar (kills on contact → predicted danger, planned around); a move that lost the game is not repeated in the level; a graph with its frontier out of reach is reopened instead of wandered | done — 23 / 110, official-style score 0.44 (from 0.42), new default |
| 13 | **The avatar's body**: sprites that move in lockstep with the avatar whenever they touch it are part of it (ls20's bar over its block); sweeps, predictions, contact rules and routes use the whole body | done — neutral (23 / 110, 0.44), kept: the footprint is now right |
| 14 | **Exploration variants under the official metric**: object-targeted exploration (Phase 8a) re-measured on the score, where a cheaper first level counts | done — no gain (0.39 vs 0.44): faster first level on `tu93`, dearer later ones; kept as `model-explore` |
| 15 | **Switches and indicators**: tile kinds whose entry changes a far-away object are switches, the objects they change are indicators, their arrangement is the avatar's state; a goal reached without a level-up locks the state and sends the agent to an untried switch | done — neutral (23 / 110, 0.44); fires only on `ls20`, where the energy budget ends the attempts first; kept |
| 16 | **Patrols in motion**: per-object velocities, a bounce model that predicts a patrol's reach for forty moves, and route search over position *and* time | done — neutral (23 / 110, 0.44; `tu93` 1.88 → 1.91), kept: the second no-gain phase in a row, loop stopped |
| 17 | **Budget-aware attempts**: with a known move budget, a plan or frontier walk that cannot finish in the attempt triggers a deliberate reset (only when a fresh attempt could reach it) | done — neutral (23 / 110, 0.44): it fires on `tu93` and `g50t` a dozen times a run and saves nothing measurable; kept |
| 18 | **Resource bars**: the status-area colour that runs out per action and reads empty at losses, the tiles that refill it, refill plans before a long route | done — neutral (23 / 110, 0.44): `ls20` learns its energy bar and refill pickups, loses the game less, and still does not reach its second level; kept — second no-gain phase in a row |
| 19 | **Competition harness dry run**: the bundle through the official framework's own loop on all 25 games, action and wall-clock caps for the threaded rerun, submission steps | done |
| 20 | **Scaling check** at 12,000 steps (3 seeds): levels per run 0.88 → 1.09, official-style score unchanged (0.39 → 0.39 on the same seeds): levels found late score nothing | done |
| 21 | **Solution replay**: the shortest click sequence that completed a level, replayed by target signature in the next | done — a loss (0.41 vs 0.44): sequences belong to one layout; off by default, kept as a variant |
| 22 | **Difference reduction**: the kinds that came together when a level ended (a moving kind, a static kind) learned as the goal relation; later levels explored best-first towards the frames where their gap is smallest | done — neutral (0.43): the frontier choice never matters in a click game, every node still has untested clicks; kept |
| 23 | **Clicks ordered by learned gain**: the kinds whose clicks have closed the goal gap go first at every node; relation kinds matched by size and shape when their colour changes between levels | done — levels 110 → 113 (`lp85` 8 → 14: two seeds reach its fifth level), score unchanged (0.44); new default; loop stopped by its rule |
| 24 | **Game families**: the 25 public games grouped by mechanic, with each family's levels, human baselines, the agent's cost and its share of the official-style score (`scripts/families.py`) | done — navigation and marker games are worth 28 points each; the agent took 0.08 and 0.24 |
| 25 | **Mechanics library, marker family**: a solver that learns what each action does to the pieces from the frames, finds the marks of each piece's colour, plans the shortest action sequence to cover them, and takes control from the model-guided agent only with a plan in hand | done — official-style score 0.44 → 4.52, levels 113 → 143; `lp85` every level on every seed; new default (`library`) |
| 26 | **Mechanics library, navigation**: the avatar as a piece — goals of another colour learned from where it ended a level, walkable and blocking colours, pieces that turn as they move | done — no gain (4.52 with and without, `library-nonav` ablation): what loses navigation levels is each game's own mechanism (spells, ghosts, carried glyphs, enemies), not the walking |
| 27 | **Hand-over discipline**: in a game the solver has not solved a level of, at most 15 probes and two selection presses a level; no strike when there is nothing to aim at yet; a take-over plans at once | done — no gain (4.52; `r11l` 1.02 → 1.22): the second no-gain phase in a row, loop stopped |
| 28 | **Mechanics library, copy family**: a solver that recognises a canvas to be painted like a reference shown beside it, learns the cursor places, brushes (completed by simple shapes), palette and stamps from the frames, and plans the strokes by peeling the reference in reverse | done — official-style score 4.52 → **9.03**, levels 141 → 164; `cd82` every level on every seed, in about 152 actions a run against the humans' 171; nothing else changes |
| 10 | (optional) a local language model that proposes hypotheses | — |

**Where this stands.** The default `model` agent beats the exploration baseline at every
budget tried, and the gap widens with the budget (levels per run, all 25 public games):

| step budget | graph exploration | model (this repo) | gain |
|---|---|---|---|
| 4,000 (5 seeds) | 0.60 | 0.88 | +47 % |
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

## Submitting to Kaggle (ARC Prize 2026, ARC-AGI-3 track)

The Kaggle notebook embeds exactly one file — `agent/my_agent.py` of the official
[ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) kit — and
runs it inside the ARC-AGI-3-Agents framework against a local gateway, with no internet.
`scripts/bundle.py` packs this whole package into that one file:

```bash
python scripts/bundle.py                      # writes kaggle/my_agent.py (also committed)
```

`kaggle/my_agent.py` embeds every module's source and materialises them as real modules
at import time, so the submitted code is byte-for-byte what the tests cover; a `MyAgent`
class runs the `library` agent (the family solvers in front of the model-guided agent),
adapts the framework's `FrameData`/`GameAction` to the engine types and never stops on
`GAME_OVER` (the agent resets the level and carries on).
`tests/test_bundle.py` runs the bundle through a stand-in for the framework's `Agent`
loop against the local engine, and the dry run below ran it through the real one.

**What the competition rerun does** (read from the starter kit's notebook builder and the
framework's `Swarm`): the notebook copies the framework next to the agent, registers
`MyAgent`, points it at a local gateway (`http://gateway:8001`, no internet) and runs
`python main.py --agent myagent`; the swarm then plays *every* game at once, one thread
per game in one Python process, and closes the scorecard only when all threads have
returned. Two consequences shape the bundle: the agent's per-game CPU time adds up
across games (the threads share one interpreter), and an agent that never stops would
run the notebook into its wall-clock limit before the scorecard is closed. So the
bundled agent stops a game at `NEO4_MAX_ACTIONS` (40,000) actions or at
`NEO4_TIME_BUDGET` (three hours) of wall clock counted from its construction, so that
all games stop together whatever their number. More actions never lower the score —
a level's score counts only the actions until it was completed — so the cap is a
safety margin, not a strategy.

**Dry run through the official framework.** The starter kit's `scripts/play_local.py`
drives `agent/my_agent.py` with the framework's own `Agent.main()` loop against the
`arc-agi` engine in-process — the same loop the gateway drives on Kaggle. With this
repository's bundle copied in, all 25 public games ran to a 1,000-action cap at 76–400
actions per second (2.5–13 s per game; on a machine busy with a sweep), with the
levels the 4,000-step sweeps predict for that budget, and the package's own scorecard
reported the result. That scorecard implements the formula this README calls the
official-style score — per level `min(115, (human / agent)² × 100)`, weighted by level
number — so the two agree.

To submit:

1. `git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter && cd ARC-AGI-3-Kaggle-Starter`
   and follow its README (`make setup`, a Kaggle token in `.kaggle/access_token`, your
   username in `notebooks/kernel-metadata.json`). It needs Python 3.12.
2. Copy `kaggle/my_agent.py` from this repository over `agent/my_agent.py`.
3. Set `ACCELERATOR = "cpu"` in `scripts/build_notebook.py` — the agent uses no GPU.
4. `make play-local` to check it plays (every game, 200 actions each), then
   `make submit`; `make status` follows the run. Kaggle first runs the notebook as a
   check, then reruns it on the hidden games when you press *Submit to Competition*.
5. If the hidden set is large or the rerun's allowance is short, lower
   `NEO4_TIME_BUDGET` (seconds per game, wall clock) at the top of the agent file; at the
   measured 100–300 actions per second, 40,000 actions take 2–7 minutes of CPU per game.

The dependencies the notebook installs offline are `arc-agi` (with `arcengine`) and
`python-dotenv`; `numpy` and `scipy` are part of the Kaggle image, and the frame code
falls back to a pure-Python labeller if `scipy` is ever missing.

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
  marker.py       the marker-family solver: learned action maps, marks, search over piece positions
  painter.py      the copy-family solver: canvas and reference, learned brushes, strokes by peeling
  library.py      the mechanics library: family solvers in front of the model-guided agent ("library")
  run.py          runner: games x seeds under a step budget, multi-process; summariser; `explain`
results/          summarised result tables (raw per-run JSON lines are git-ignored)
scripts/bundle.py builds the single-file Kaggle agent
scripts/families.py the public games by mechanic family, with each family's share of the score
kaggle/           my_agent.py — the generated submission file (do not edit by hand)
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
  and which click targets never change anything. A move is a *sweep*: every cell between
  the avatar and its destination counts as crossed, so an avatar that jumps six cells
  learns that the wall it never lands on is what stops it.
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
  tested first. When no route reaches the goal sprite, the plan aims at the sprite that
  encloses it (a box with the exit pattern inside), then at colour goals that are not
  walls elsewhere.
* **Losses explained before they are blamed.** A level whose losses fall at the same
  attempt length from different places has a move budget, and those losses credit
  nothing. The frame of a lost game names the kind that moved into the avatar's place;
  such a kind kills on contact, and its reach — one avatar step around it — is avoided.
  A kind in motion is expected where its motion carries it: its last displacement is
  extrapolated with a bounce model (keep going, reverse at cells the avatar cannot
  cross), and the planner searches over position *and time* so a route crosses a
  patrol's corridor when the patrol is elsewhere. A move that lost the game is not
  repeated in the level.
* **Switches and indicators.** A tile kind whose entry changes the same far-away object
  nearly every time is a switch; the objects switches change are indicators, and their
  arrangement is the avatar's *state*. A goal plan that reaches the goal without ending
  the level locks that state; the agent then plans to a switch not yet tried from the
  state and plans for the goal again.

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

## Results — Phase 8c (adjacency goal conditions): neutral

Adjacency between the avatar and every other object is now recorded on each keyboard
action, including ones that do not move the avatar. Objects beside the avatar when a level
ended, beyond chance, become goal conditions: the planner targets the band around them
when no entry-goal object is on screen, and once beside one the key that completed levels
before (or a move towards the object) is tested first. On a toy room where a level ends by
pressing a key next to a block this works as intended (later levels in a fraction of the
steps of the first). On the public games it is neutral — **20 / 97** against 21 / 97 —
because the conditions it can express rarely hold: in `tu93` the avatar is beside a block
on 3,327 of 4,000 steps, so adjacency carries no information there; the level ends when
a *block* reaches a *target*, a relation between two objects other than the avatar. Off
by default; `model-adj` keeps it.

## Results — Phase 10 (robust avatar identification): neutral in levels, right in substance

A deep look at `tu93` (its public source is in the toolkit's download) found a flaw in the
object model rather than in the game logic. The player sprite carries an orientation
marker — an "eye" pixel that turns with it — so its exact shape hash changed every time
it turned and the matcher never saw it move; meanwhile the identical floor tiles it
covered appeared to move by the opposite vector, and the avatar detector chose a floor
tile (29 copies of it on screen). Two general fixes:

* a sprite's shape hash is taken over the canonical member of its eight rotations and
  mirrorings, and the signature's size fields are orientation-free (`sig`); click rules,
  where a target's orientation can be what it means, keep an exact signature (`esig`) —
  the first sweep without that distinction lost levels on `r11l` and `cd82`;
* the avatar score divides observed moves by the mean number of instances of the
  signature per frame: the avatar is usually the one object of its kind.

On `tu93` the agent now identifies the colour-9 player with the right four directions,
learns `entering the colour-14 3x3 sprite completed a level (3/4 contacts)` and plans to
it (5–10 plans per run, none failing); the first level falls in 197 steps instead of 407.
Over the 25 games the result is **20 / 95** against 21 / 97 — within run-to-run noise
(`dc22` +1, `tu93` −2 over seeds, `tr87` −1). It is kept as the default because every
avatar-dependent component (pruning, hazards, goals, planning) now reasons about the
right object. Two hypotheses were checked on the same data and dropped before building:
"an object kind approaches the avatar when the game is lost" (no kind is near the avatar
at death beyond chance in any game; `tu93`'s losses are its per-level step limit) and
the push-chain pattern (avatar → block → target) for `tu93`.

## Results — Phase 11 (swept-cell world model): planning fires on `tu93`, +0.03 official

Why did the planner never fire on `tu93`'s later levels although the goal sprite was
known? Its avatar moves six cells per action, and the model credited a stopped move to
the cells the avatar *would have landed on*: the floor colour collected 24 "blocked"
counts against 171 "passed", stayed below the 90 % pass rate, no cell was passable, and
every route search failed. Three general changes:

* **Sweeps.** A move passes through every cell between the avatar and its destination
  (`planner.swept`, nearest band first). A successful move credits all swept colours as
  passed; a stopped move blames the swept colours not yet known to be crossable — early
  on the floor too, until its passes establish it. Prediction and route search check the
  whole sweep, so a jump over a thin wall is "blocked" even though its landing cells are
  floor.
* **Budget deaths do not kill edges.** `tu93` ends every attempt after fifty moves. A loss
  recorded on a well-trodden graph edge turned it into a dead end; with the reverse link
  still in place, path finding kept routing through it, `edge_to` found nothing, and the
  agent reset on every visit (one seed spent 3,900 of 4,000 steps that way). An edge that
  led somewhere more often than it killed now survives a loss; an edge that is downgraded
  drops its reverse link.
* **Blame by suspicion.** Blocked credit goes only to swept colours that are not already
  established as crossable, which keeps a well-trodden floor from accumulating blocks when
  an object stands in the way.

Over the 25 games (4,000 steps, 5 seeds): **21 / 98** levels against 20 / 95, and the
official-style score **0.42** against 0.39 — the whole gain is `tu93` (0.73 → 1.55),
whose second level now falls in 14 actions on one seed (the human baseline is 16: the
level scores the 115 cap). Everything else is unchanged to the second decimal, which is
what a fix to the model of one movement style should look like.

What still fails on `tu93` is instructive. Its later levels have patrolling enemies that
kill on contact; a planned route walks into one, the attempt ends, the level restarts and
the agent makes *the same plan* again — five times, the per-level plan allowance — because
nothing in the model ties a death to a moving object, and the budget deaths dilute every
colour-level hazard count. And once the enemies are in the state key, the level graph
explodes (477 nodes on one seed) and the frontier becomes unreachable after a reset. Both
point at the next phase: deaths explained by what *moved into* the avatar, a level-local
memory of fatal moves for the planner, and a budget rule that stops counting the
fifty-move deaths as hazards.

## Results — Phase 12 (death attribution): +12 levels, official-style 0.44

A loss is now explained before it is blamed. Four changes, all general:

* **Move budgets.** When at least two losses in a level fell at the longest attempt
  length seen in that level, *from different places*, the level has a move budget. A
  loss at that length credits no hazard colour, no dangerous action, no dead graph edge:
  the edge it fell on is left untested. (The "different places" clause matters: one
  fatal move repeated at the same point of every attempt looked like a budget at first,
  and the agent then walked into it eighty times.) `tu93` reads 50 on its first levels
  and 35 and 45 later; `sp80` reads 45; `cd82` — a click game — turns out to have one too.
* **Kinds that kill on contact.** The frame of a lost game is no longer thrown away: it
  usually still shows the scene, with whatever killed the avatar standing where the
  avatar went. An object kind that *moved into* the avatar's place at a loss (not the
  floor it uncovered by vanishing, not something that already stood there), far more
  often than it is beside the avatar otherwise, kills on contact. A move into its reach
  — one avatar step around it — is predicted dangerous, and the planner routes around
  that reach. On `tu93` this names the colour-8 patrol and nothing else.
* **Fatal-move memory.** The (position, action) of a move that lost the game is kept for
  the level: the planner never takes it and the chooser skips it in the first epoch.
  The second plan on `tu93`'s third level differs from the first; the first used to be
  repeated five times.
* **Reopen when stuck.** Eight random walks in a row (the frontier is known but out of
  reach: a one-way door, a patrol that kept killing on the way) reopen the graph instead
  of wandering for the rest of the budget. One `tu93` seed had spent 3,900 of its 4,000
  steps wandering.

Over the 25 games (4,000 steps, 5 seeds): **23 / 110** levels against 21 / 98, and the
official-style score **0.44** against 0.42. `tu93` completes four levels on every seed
(12 → 20 levels over the seeds; its third level, the one whose patrol killed every plan,
now falls in 500–900 actions instead of never); `sp80` gains three levels over the seeds
and `cd82` one; no game loses a level, and no game's score moves by more than 0.03
except `tu93`'s (1.55 → 1.88).

What the per-seed numbers say about the remaining cost: on `tu93` a run still loses the
game about 115 times, 80 of them to the budget. The patrol's reach is a coarse model of
a patrol that only charges when the avatar is in its line; the agent gets past it by
re-testing after the graph reopens, not by understanding the trigger. A relational
rule ("that kind charges along its row or column") is the kind of thing a language
model could propose and the data could check — the plan's optional phase.

## Results — Phase 13 (the avatar's body): neutral, kept

Why do goal plans come late or never on `dc22` and `ls20`, although the goal sprite is
known and on screen? A sampling of the planner's preconditions on both games (avatar
found, directions known, goal on screen, passable colours, route) showed the same
symptom — no route — with different causes.

On `ls20` the avatar is a colour-9 block with a colour-12 bar riding on top of it; the
keyboard was seen to move the block, so the bar was just another object, and the body's
true footprint was wrong. Sprites touching the avatar that move by its vector whenever
it moves are now its *companions* (the bar: 1,142 of 1,142 moves), and the avatar plus
the companions touching it form one body for sweeps, predictions, contact rules and
route planning. That is right in substance and neutral in result: **23 / 110** levels
and **0.44**, the same as Phase 12 (`lf52` +1 and `sk48` −1 over the seeds, noise).

The route on `ls20` is still missing for another reason, and its public source says
why: the avatar carries a pattern that tiles on the floor cycle (shape, colour,
rotation), and the box that completes the level admits the avatar only when the carried
pattern matches the box's — a precondition on the avatar's *state*, not on the map.
`dc22` is a tile puzzle: the avatar walks only on tiles, and clicks slide blocks to
build the path to the goal. Neither is a planning defect; both are mechanics that a
model over colours, signatures and positions cannot express yet.

## Results — Phase 14 (exploration variants under the official metric): no gain

Phase 8a's object-targeted exploration gave faster first levels and was set aside
because it completed no extra levels — a judgement made on levels, before the official
score was computed. Under the score a cheaper first level counts, so the variant was
re-run with the current code (`model-explore`, 4,000 steps, 5 seeds). It is identical
to the default on 23 of 25 games to the step: its explore plans only ever fire on `tu93`
and `ls20`, since elsewhere the frontier is never out of unknowns. On `tu93` it halves
the first level (121 actions against 238) and then pays for it on every later level,
where walking to untouched objects competes with the goal plan: 0.71 against 1.88 on
that game, and **0.39 against 0.44** overall, with the same 111 / 110 levels. Not
promoted.

This is the second phase in a row without a gain on the score, which is where the
development loop was set to stop. Where things stand is below.

## Results — Phase 15 (switches and indicators): neutral

The mechanism for "collect the key, then the door" in its general form. A tile kind whose
entry changes the same far-away object nearly every time it is entered is a *switch*
(`ls20`: two floor tiles move a colour-9 piece in the pattern panel, 87 of 87 entries);
the objects switches change are *indicators*, and the arrangement of their exact
signatures and positions is the avatar's *state*. A goal plan that reaches the goal and
does not end the level locks that state; the agent then plans a route to a switch it has
not tried from this state, and plans for the goal again once the state has changed. Goal
plans also learned to aim at the sprite *enclosing* an unreachable goal (the box around
`ls20`'s exit pattern) before falling back to colour goals that are not walls elsewhere.
On a toy room with a three-state switch the second level falls in a few dozen actions
where exploration needs thousands (`tests/test_switch.py`).

On the public games it is exactly neutral — **23 / 110**, **0.44** — because it fires only
on `ls20` (14–30 switch plans per run), and there every attempt at the box ends with the
energy bar running out on the way (16–21 moves to the box, 19–27 losses per run). The
level needs the energy mechanic (what refills it, and when to go for it) before the
state mechanic can pay. No other public game shows a switch, so the cost is nil and the
mechanism is kept.

## Results — Phase 16 (patrols in motion): neutral

`tu93`'s later levels hold two kinds of enemy. One patrols its corridor without pause:
a bounce model — keep the last displacement, reverse it at cells the avatar cannot cross
— predicts 74 % of its steps from the recorded frames. The other stands still until the
avatar is in its line at a set distance, then charges; it moves on 2 of 699 steps that
look "aligned" to a bounding-box test, so nothing simpler than the game's own rule
predicts it. The model now tracks every object's last displacement, extrapolates the
reach of hazard kinds in motion for forty moves, and the planner searches over
(position, time) while the prediction lasts (`planner.plan_path(avoid_t=...)`), so a
route crosses a patrol's corridor when the patrol is elsewhere; a kind at rest keeps its
static reach.

Result: **23 / 110** levels and **0.44**, with `tu93` 1.88 → 1.91 — one seed's fourth
level falls in 304 actions instead of 1,035, the other four seeds are unchanged to the
action, and the fifth level stays out of reach because the static chargers' reach (one
avatar step around them) still walls off the corridor the route needs. The mechanism is
right where it applies and is kept; it is the second phase in a row without a gain on the
score, which is where this loop stops.

**Where this stands after Phases 15–20.** The default agent is unchanged in result since
Phase 12: 23 median levels, 110 over all runs, official-style score 0.44 at 4,000 steps,
1.09 levels per run at 12,000. Phases 17 and 18 (budget-aware attempts, resource bars)
were neutral like 15 and 16, each right where it applies and each blocked by the next
mechanic in the same game; Phase 19 validated the bundle through the official
framework's own loop and gave it the caps the threaded competition rerun needs. The two
mechanics that these phases model — a state the avatar must be in, and hazards that move
— are each implemented, tested on toy rooms, and blocked on the public games by a
*third* mechanic in the same game (`ls20`'s energy bar; `tu93`'s charge trigger). That is
the shape of the remaining problem: the public games compose three or four mechanics
each, and a hand-built model gains a level only when every mechanic on the route is
expressed. The next step that does not have that shape is a library of mechanics learned
across games, or a language model proposing the composition for the data to check.

## Results — Phase 17 (budget-aware attempts): neutral

With a level's move budget known (Phase 12), a plan or a walk to the nearest untested
edge that cannot finish within the actions left in the attempt is not started: the level
is reset instead — one action for a fresh budget, where the walk would have spent them
all. A route longer than any attempt can be is not planned at all. The first version
reset whenever the frontier was out of reach, which on `ls20` meant a reset on every
visit and no progress; a reset is now taken only when a fresh attempt could reach the
frontier from the start.

Over the 25 games: **23 / 110** levels and **0.44** (0.436 against 0.437), identical to
the step on every game but `tu93`, where it is within noise. The rule fires a dozen
times a run on `tu93` and thirty on `g50t`, and saves nothing measurable: on `tu93` the
budget is 35–50 actions and the frontier is rarely farther than that, so the walks it
prevents are short, and in click games every node still has untested clicks, so the rule
never applies. Kept, as it is right and costs nothing. First scoring phase of this
loop without a gain.

## Results — Phase 18 (resource bars): neutral

`ls20`'s losses were unexplained because its energy bar lives in the status area, which
the state key masks out. The model now reads that area: a colour whose pixel count falls
by a fixed amount on most actions and reads lowest when the game is lost is a *resource*
(`ls20`: colour 11, 4 pixels per action, at its lowest on 8 of 8 losses — including the
losses the budget rule had classed as budget losses, which are kept for this purpose and
a colour that has run out is read as zero), the count divided by the drop is the actions
it still pays for, and tile kinds whose entry makes it jump back up are *refills*
(`ls20`'s colour-11 pickups). A plan the resource cannot pay for sends the agent to a
refill tile within reach; a reset is never taken for energy, since a level restart
restores the bar but not the position. On a toy room with a bar and a pickup the second
level falls in a few dozen actions where the direct route is impossible
(`tests/test_energy.py`).

Over the 25 games: **23 / 110** and **0.44** (0.436), the same as Phases 16 and 17.
On `ls20` the refill plans fire (1–8 per run) and the losses per run fall from about 35 to
22–48 depending on the seed, but the second level stays out of reach: its pickups are
one-time, the cycle of "reach the box, find it locked, walk to a switch, come back"
costs more than one bar, and the agent cannot yet ration the pickups across that cycle.
Second scoring phase in a row without a gain: by the loop's rule, development stops
here, and the last item of the plan — the scaling check — is a measurement.

## Results — Phase 20 (scaling check at 12,000 steps)

The default agent, three seeds, 12,000 steps per game (`results/raw/model@12k-p18.jsonl`),
against the same seeds at 4,000 steps:

| budget | levels per run | official-style score (seeds 0–2) |
|---|---|---|
| 4,000 | 0.88 | 0.39 |
| 12,000 | 1.09 | 0.39 |
| 12,000, Phase 7 agent (old run) | 0.89 | 0.30 |
| 12,000, graph exploration (old run) | 0.69 | 0.24 |

A larger budget adds levels (+24 %: a fifth `tu93` level on two seeds, a third level
on `dc22`, `vc33` and `sp80`) and **no score at all** — 0.394 to 0.394 on the same seeds.
The levels the extra budget buys are found by exploration after thousands of actions,
and a level completed in thousands of actions against a human's tens is worth a few
hundredths of a point. This is the metric's verdict on the whole approach, stated
plainly: more time does not help; only levels completed cheaply count, and those come
from a model that understands the level, not from a longer search. At the same budget
the current agent completes 22 % more levels than the Phase 7 agent and 58 % more than
exploration, which is the "beyond" strand's measure, not the competition's. For a
submission the action cap still costs nothing as long as the wall clock allows it.

## Results — Phase 21 (solution replay): a loss, off by default

The hypothesis: the later levels of a click game repeat its mechanic with a new layout,
so the sequence that completed one level should be a good first try at the next. Every
click candidate now carries its target's signature (colour, size, exact shape); at a
level-up the shortest known sequence from the level's first frame to its completing
move is extracted from the level graph, and in the next level it is replayed, each click
going to the nearest target of the recorded signature, aborting at the first missing
target or no-change step. On a toy room with three buttons to press in order the second
and third levels fall in three clicks each (`tests/test_replay.py`).

On the public games it transfers nothing: `lp85`'s sequence is 37 clicks that belong to
its first layout; `vc33`'s is the number of presses one layout needed; `r11l`'s first
target changes shape after it is clicked. Over the sweep **0.41 against 0.44** (109 /
110 levels): `vc33` loses 0.5 because the replay's wasted clicks are paid for on its
second level, the one it completes cheaply. Off by default (`max_replays = 0`), kept as
a variant. What the accounting behind it showed is the real finding: in a click game's
later levels nearly every click produces a new state (2,500 distinct states in a
4,000-step run), so raw exploration of the state graph cannot finish — the search needs
a direction, which is what Phase 22 gives it.

## Results — Phase 22 (difference reduction): neutral, and a lesson about where search goes

The planned "soft replay" was dropped once Phase 21 showed sequences do not transfer;
what the click games need is a direction for their search. At a level-up the agent
now looks for the pair of kinds that came together: a kind whose instances moved (or
appeared) and ended nearest to a kind that stood still, ranked by how steadily their
bounding-box gap closed along the solution path through the level graph, then by how
much it closed. The two best pairs are the *goal relation*; every later node carries
the gap between the nearest such pair, and the walk to the frontier aims at the nodes
with the smallest gap instead of the nearest. On a toy slide puzzle (a block pushed
towards a target by a button, a decoy that does nothing) the second level is a straight
descent (`tests/test_reduction.py`).

On the public games: **0.434 against 0.436**, 107 / 110 levels — and, to the action,
the same second-level timings on `vc33`, `r11l` and `lp85` in every seed. The reason is
plain in the traces: in a click game every node still has untested clicks (dozens of
targets), so the agent never walks to a frontier at all, and which frontier the
heuristic would prefer never comes into play. The 36-action second level seen on `vc33`
during development came from a different component — ordering a node's clicks by
proximity to the moving kind — which helped `vc33` and hurt `r11l` and `lp85`; it is
off by default. The lesson is the next phase: in click games the search is decided by
*which click to try first at a node*, so that is where the learned relation must act.

## Results — Phase 23 (clicks ordered by learned gain): more levels, same score

Phase 22's lesson applied: in a click game the cost of a level is decided by which
click is tried first at each node. Every click now credits its target kind with the
change it produced in the goal-relation gap; at a node, clicks on kinds whose mean
change is negative go first, kinds not yet tried next, kinds that never closed the gap
last. Relation kinds are matched by size and shape when their exact colour is absent,
since levels recolour their pieces (`vc33`'s markers are colour 11 on one level and 14
on the next). On the toy slide puzzle the second level now takes exactly its seventeen
pushes (`tests/test_reduction.py`).

Over the 25 games: **113 levels** over all runs against 110, the first rise since Phase
12, all of it `lp85` (8 → 14: two seeds now complete five of its eight levels, the
third to fifth in 360–570 actions each with 460–2,600 gain-ordered clicks) — and the
official-style score unchanged at **0.44** (1.63 → 1.68 on `lp85`, since 400 actions
against a human's 16–41 is worth little; `tu93` and `cd82` move within noise). On
`vc33` and `r11l` the ordering fires and changes nothing: their relation is not the
one that matters, and the gap it measures does not move with the clicks that do.

**Where this stands after Phases 21–23.** The default agent completes 23 median levels
and 113 over all runs at 4,000 steps, official-style score 0.44 (exploration baseline:
17 / 75, 0.26). Three phases in a row without a score gain is the loop's stop rule, and
it stops here. What the three phases established: sequences do not transfer between
levels (Phase 21); in click games the search never reaches a frontier, so only the
local click order matters (Phase 22); a learned "which kind closes the gap" order does
win levels where the relation is right (`lp85`) and nothing where it is not (Phase 23).
The relation is the bottleneck: it is guessed from one level-up by geometry, and in
`vc33` and `r11l` the true goal is not a gap between two kinds. Learning what a
level's goal *is* — from the frames and actions of a completed level, in a form that
can be checked against the next level — is the open problem that separates this agent
from a strong one, and it is the problem a language model proposing hypotheses for the
data to check would be aimed at.

## The mechanics library (Phases 24–)

The metric pays for levels finished in close to a human's number of actions, and an
explorer that starts from nothing cannot get there: every level it finishes was paid
for in exploration. What a human brings to a new game is a set of expectations about
what games are like — a piece is meant to go on the mark of its colour, a wall stops
you, a key opens a door. The library is an attempt to give the agent such expectations
without a language model (the competition runs offline on Kaggle's CPU or T4 hardware,
with no internet): for each *family* of mechanics, a solver that recognises its family
from what it sees, plans in a model of that family, and gives the game back to the
model-guided agent the moment the game stops behaving as the family predicts.

The solvers never see a game's identity or its source. The grouping below was made by
reading the public games' sources, which is fair for choosing what to build and unfair
for measuring it: the evaluation games are different games, so gains on the public set
are an upper bound on what transfers. The solvers are written to depend only on what is
on screen, and are tested on small synthetic games that share a family with no public
game's layout, colours or sizes.

### Phase 24 — the public games by family

`scripts/families.py` prints this table from the result files. "Agent actions / human"
is the ratio of the agent's actions to the human baseline on the levels it completes;
"contribution" is the family's part of the 25-game official-style mean (0–100); a family
of *n* games can contribute at most about 4*n* points.

| Family | Games | Levels | Human actions | Levels completed (model agent, mean of 5 seeds) | Agent / human | Contribution now | At most |
|---|---|---:|---:|---:|---:|---:|---:|
| Navigate an avatar to an exit (keys, switches, hazards, budgets) | ls20, tu93, sc25, g50t, dc22, bp35, m0r0 | 50 | 5,453 | 8.0 | 15.8× | 0.08 | 28 |
| Bring pieces onto markers by indirect control (no avatar) | lp85, re86, s5i5, r11l, vc33, ar25, su15 | 54 | 4,070 | 9.6 | 21.8× | 0.24 | 28 |
| Push or carry blocks into marked zones | wa30, ka59 | 16 | 2,573 | 0.0 | – | 0.00 | 8 |
| Reproduce a shown pattern or sequence | cd82, tr87, sk48 | 20 | 1,655 | 1.4 | 23.0× | 0.01 | 12 |
| Satisfy local rules on a board (toggles, pegs, jigsaw) | ft09, lf52, cn04 | 22 | 2,336 | 2.0 | 32.9× | 0.00 | 12 |
| Compose a program or route a flow, then run it | tn36, sb26, sp80 | 21 | 1,048 | 1.6 | 7.3× | 0.10 | 12 |

The games, in a line each (what the player does; what ends a level; what loses it):

| Game | Family | How it is played |
|---|---|---|
| ls20 | navigate | A 5×5 avatar walks a maze in 5-px steps; the exit pad admits it only while the glyph it carries matches the pad's (shape, colour and rotation are changed by stepping on tiles); a 42-step energy bar with refills, three lives. |
| tu93 | navigate | A 3×3 avatar on a lattice maze, one cell per key; reach the exit tile under a per-level move budget, past patrols that bounce, chargers that run along a line once armed, and followers that copy the avatar's path. |
| sc25 | navigate | Walk a maze to the exit; barriers fall, the avatar shrinks or teleports when a 3×3 grid is clicked into a spell's pattern; a move budget with refills. |
| g50t | navigate | Walk to the exit bracket; pressure plates hold doors open only while occupied, so a run is recorded and replayed as a ghost that holds the plate; a timer. |
| dc22 | navigate | A 2×2 avatar must always stand on floor; panel buttons swap which floor pieces are solid, keys reveal buttons, a crane carries bridges; a falling avatar costs 20 steps. |
| bp35 | navigate | A side-view avatar steps and falls; clicks dig blocks, toggle platforms and flip gravity on the way to a gem; spikes, a rising flood, an action budget. |
| m0r0 | navigate | Two dots move as mirror images; walls desynchronise them until they meet; switches open gates of their colour, hazards reset positions. |
| lp85 | marker | Arrow buttons turn rings of tiles; the goal tiles must stop inside the corner marks of their colour; rings intersect in later levels. |
| re86 | marker | Arrow keys move the selected line drawing (a cross, a frame) in 3-px steps, a key selects the next; every dot must be covered by a stroke of its colour; paint pots recolour and obstacles bend pieces later on. |
| s5i5 | marker | Clicking a slider's halves lengthens or shortens every arm of its colour; each arm's tip must reach a cross mark; arms carry child arms, rotate buttons appear later. |
| r11l | marker | Click a handle, then a destination: the piece sits at the centroid of its handles and must overlap the ring of its colour, off the hazard blobs; later, blank pieces collect colours from pickups. |
| vc33 | marker | Buttons raise one column and lower its partner; tokens ride the columns to the height of the wall patch of their colour, and swap columns over a bridge when two columns are level. |
| ar25 | marker | Move a piece or a mirror line; the piece and its reflections must cover every target dot. |
| su15 | marker | A click pulls nearby pieces together; equal pieces merge into the next size; pieces of the required size must end inside the target rings, away from the crabs that bite. |
| wa30 | sokoban | A 4×4 avatar grabs, drags and drops blocks into outlined zones; helper robots carry blocks in, thief robots carry them out to traps. |
| ka59 | sokoban | Select a block and walk it; walking into another block kicks it sliding (further on ice); blocks must fill outline slots of their size; timed cannons shove blocks. |
| cd82 | copy | Paint a canvas to match a reference by stamping halves and triangles of the selected colour from eight positions round it, in the right order. |
| tr87 | copy | Rewrite rules are shown as tile pairs; set each output tile (cycling glyphs) to the translation of the input row; later levels edit the rules instead. |
| sk48 | copy | Extend, retract and slide a chain to pick up beads so the colours along it match a template chain; undo is free. |
| ft09 | constraint | Clicking a tile cycles its colour (some tiles also flip their neighbours); every constraint glyph must see its neighbours equal to or different from its centre colour. |
| lf52 | constraint | Peg solitaire: jump a peg over a neighbour to remove it, one peg (two in some levels) must remain; sliding platforms carry pegs between islands. |
| cn04 | constraint | Move and rotate jigsaw pieces until every connector pixel meets a matching one; later levels hide unselected pieces' colours and stack variants. |
| tn36 | program | Set the switches of a program strip and run it to drive a robot onto a bracket in the right pose, scale and colour; a demo panel shows what the opcodes do. |
| sb26 | program | Place colour squares into a program frame (hollow squares call sub-frames) so that running it paints the target sequence. |
| sp80 | program | Move platforms and deflectors, then pour: every cup must fill and no drop may reach the floor; four failed pours lose. |

The two largest families are worth 28 points each; the agent takes 0.08 and 0.24 of
them. The marker family is attacked first: its games share one goal structure that is
visible on screen (a piece and a mark of the same colour), so one solver can serve all
of it, where the navigation games share an avatar but differ in what blocks, kills and
unlocks.

### Phase 25 — the marker family: pieces onto the mark of their colour

`neo4/marker.py` holds one solver for the whole family; `neo4/library.py` runs it in
front of the model-guided agent (`LibraryAgent`, the `library` agent of `neo4 run` and
the agent in the Kaggle bundle). The solver watches every transition while the
model-guided agent explores and learns three things from the frames alone.

* **What moves.** Components are matched between frames by colour and exact shape with a
  minimum-path-length assignment, so a ring of identical tiles turning by one cell reads
  as one step per tile, and a ring wrapping round reads as one long step rather than a
  reshuffle of its neighbours. A new component touching a moved one of its colour moved
  with it (an arm drawn shorter where it passes behind something). One copy of common
  scenery changing place, and two components swapping places with opposite moves, are
  occlusions (a floor cell hidden by a passing avatar), not moves; a transition that
  shifts most of the screen alike is a camera scroll.
* **What each action does.** An action is kept as a *map of places*: for a component of
  a given shape at *p*, the set of places *q* where, after the action, the colour that was
  at *p* showed up. Every observation intersects the sets; a one-to-one, least-path
  assignment over them is the current guess, so a ring guessed from identical tiles
  still closes into a cycle. An action whose moved components all moved alike is also a
  *translation* of the piece it moved, which carries on beyond the places seen (a
  selected piece, an arm tip). An action after which pieces change outline without
  moving is a *selection change*; the actions bound to those pieces go stale and are
  re-bound to whatever moves next.
* **What the goal is.** Marks are small static clusters of a colour that also moves (four
  corner dots, a cross, a dot, a ring). A mark is met when a piece of its colour covers its
  centre — tested in the current frame, so a piece that merges with its mark (an arm tip
  entering a cross) still counts. When a level ends, the first frame of the level-up
  still shows the completed level: how each mark sat relative to the nearest piece of its
  colour (on it, or beside it) is recorded by piece size, and later levels find their
  marks by that arrangement — rescaled when a level is drawn at another scale, and chosen
  as an exact cover so that marks drawn side by side on a lattice stay apart.

It plans by breadth-first search over piece positions in the learned model, for the
shortest action sequence after which every mark is met. When the model is unsure what an
action does at a place, an optimistic search lets any of the nearest candidates happen;
running such a plan is what resolves the doubt. When the joint goal is out of reach, it
covers one more mark at a time. Before a long plan it probes look-alike controls it has
not tried (controls come in families, such as the two arrows of a ring), and when no
piece of an unmet colour can move it presses the selection action and re-probes a move.
It takes control only with a plan in hand in a game it has not solved a level of, and
gives a level back after twenty mispredictions, three lost attempts or 400 actions; two
give-ups end its part in a game (three once it has solved a level).

Three small synthetic games in `tests/test_marker.py` exercise one mechanism each (a
ring turned by two buttons, an arm extended by a button whose tip merges into its mark,
two crosses moved by keys with a selection key between them); none shares a layout,
colour or size with a public game.

**Results (Phase 25, `library-nonav`: the solver without the Phase 26 additions; 4,000
steps, 5 seeds).** The official-style mean over the 25 games goes from 0.44 to **4.52**,
levels from 113 to 143 over all runs; the marker family's share from 0.24 to 4.32, and its
cost on the levels it completes from 21.8 to 6.6 times the human count.

| game | model agent | library | levels (all seeds) | median actions per level, solver vs human |
|---|---:|---:|---:|---|
| lp85 | 1.68 | **101.51** | 14 → 40 (every level, every seed) | 20/17, 18/38, 28/31, 40/16, 15/41, 23/60, 25/26, 20/159 |
| s5i5 | 0.00 | 1.63 | 5 → 5 | first level 25/20 |
| re86 | 0.00 | 1.50 | 0 → 3 | first level 26/26 (three seeds of five) |
| vc33 | 2.04 | 2.41 | 10 → 10 | 15/7, 85/18 |
| r11l | 2.34 | 1.02 | 9 → 6 | a failed take-over spends the level's 60 actions probing; second level lost on three seeds |
| other 20 | 0.40 total | 0.42 total | 75 → 79 | the solver does not take control, or gives back after a few actions |

`lp85` is the family in its purest form — the solver plays all eight levels in about 190
actions where people need 388. On `re86` and `s5i5` it matches people on the first level
and stops at the second, where the mechanics outgrow its model (`re86`'s pots recolour
pieces; `s5i5`'s arms collide with walls and carry child arms). `r11l` is a loss: its
handles move to wherever the player clicks, which a learned map of places cannot express,
and the solver probes until the level's move budget runs out.

### Phase 26 — navigation: the avatar as a piece

The navigation family is worth as much as the marker family (seven games, 28 points) and
the agent took 0.08 of it. Its games share an avatar but little else: what opens a door
is a carried glyph in `ls20`, a spell in `sc25`, a recorded ghost on a plate in `g50t`, a
panel button in `dc22`. Phase 26 does not try to model those; it extends the marker
solver so that an avatar is one more piece, which is the part the family has in common.
Three additions, all switched off in the `library-nonav` ablation:

* **Goals of another colour.** When no mark of a moved piece's own colour explains how a
  level ended, the small static object it overlapped most in the completed level's last
  frame is taken as its goal (`tu93`'s exit under its avatar), and later levels look for
  that object.
* **Walkable and blocking colours.** For a piece moved by translation, the colours it has
  stepped onto are walkable and the colours that stopped it are walls; the search routes
  translations through walkable colours (and through colours not seen yet), and checks
  the place half-way for pieces that jump a cell.
* **Pieces that turn as they move** (an avatar that faces its way) are matched by colour
  and kind when their exact shape changes.

The phase also made the library safe to leave on everywhere. The model-guided agent now
records the solver's actions as its own transitions, so a hand-over never leaves it with a
broken state graph or move-budget count (on `ls20` a twelve-action take-over that failed
had cost it a thousand actions). Planning while the other agent plays backs off after
failures and is budgeted in predictions per call and per level, so no public game runs
slower than about 20 ms a step.

**Results (Phase 26, `library` against `library-nonav`, 4,000 steps, 5 seeds): no gain.**
The official-style mean is 4.52 with and without the additions; every navigation game
scores within a hundredth of the ablation. The additions fire — on `tu93` the exit is
learned as a goal of another colour when the first level ends, and the solver walks the
avatar to it on the second — but the second level is already cheap for the model-guided
agent, and from the third level on its patrols and chargers kill the avatar (three lost
attempts give the level back). On the six other navigation games the first level is never
completed by walking alone: it needs a spell (`sc25`), a recorded ghost on a pressure
plate (`g50t`), a carried glyph that matches the exit (`ls20`), panel buttons that remake
the floor (`dc22`), digging and gravity switches (`bp35`), or two mirrored dots brought
together (`m0r0`). The family's common part — an avatar walking to a goal — is not where
its levels are lost; what is lost is each game's own mechanism, which a family solver
cannot know in advance.


### Phase 27 — hand-over discipline

Phase 25's one loss was `r11l`, where a take-over whose plan failed went on probing until
the level's move budget ran out. Phase 27 caps the solver in games it has not solved a
level of (15 probes and two selection presses a level), lets it hand the turn back without
a strike when no piece of a marked colour has moved yet (at the start of `re86`'s second
level it had given the level up a step before its marks became visible), and resets its
planning back-off at every take-over. The mean stays at **4.52** (`r11l` 1.02 → 1.22,
nothing else moves by more than a hundredth): the second phase in a row without a
measured gain, so the loop stopped here.

### Phase 28 — the copy family: paint the canvas until it matches the reference

The marker solver pays where a game's goal is on screen. The copy family shows its goal
even more plainly: a small picture, and elsewhere a blank canvas of the same size that
the player must make into a copy of it. `neo4/painter.py` is a solver for that goal that
learns the painting tools from the frames:

* **Recognition.** Two isolated rectangles of the same size (no background inside,
  background all round) with different content are a candidate picture and canvas; the
  canvas is the one an action paints, i.e. turns a set of its pixels into a single
  colour. Until something paints, the solver does nothing.
* **The cursor and its places.** The largest non-background shape in a margin round the
  canvas is the cursor; its outline relative to the canvas is a *cursor place*. Actions
  that change it without painting are moves, learned as a graph over places, so they carry
  over to every level where the canvas sits alike.
* **Brushes, completed by shape.** An action that paints at a place is a brush. One use
  shows the pixels it changed (inside its region) and the pixels it left although they had
  another colour (outside); pixels that already had the colour stay unknown. The region is
  completed with simple shapes — the axis-aligned rectangles and the corner triangles of a
  square canvas that contain everything seen inside and nothing seen outside: what every
  fitting shape covers is sure, what only some cover is uncertain. A second use in another
  colour usually settles it.
* **The palette.** A click that changes the frame (the status bar left out) but no canvas
  pixel and no cursor place selects a colour, the clicked pixel's.
* **Planning by peeling.** The stroke painted last is a brush whose sure region, among the
  pixels not yet explained, is one colour in the picture (and whose uncertain pixels would
  not spoil it); those pixels are explained, and the search goes on until what is left
  already matches the canvas. Strokes that change nothing in the end are dropped; the
  cursor is routed over the learned graph, colours are selected on the way, and the plan is
  rebuilt after every action.
* **Pixels the game does not check.** When no exact plan exists, a plan that leaves at most
  a line's worth of pixels wrong is tried; if the canvas then matches all but those pixels
  and the level goes on, they are marked as checked. Pixels that differed from the picture
  when a level was completed are left out at later levels.
* **What it does not know yet, it surveys.** On the first level it visits every cursor place
  it can reach and tries every brush once, in the colour the canvas has least of, before
  painting the picture: later levels need brushes the first does not, and a first-level
  action costs the least weight. When a click next to the cursor paints for the first time
  (a stamp), it looks for stamps at every place before planning.

The library now gives the turn to the solver in control first, so a solver halfway through a
plan is not interrupted by another that recognises something.

**Results (Phase 28, `library`, 4,000 steps, 5 seeds): official-style mean 4.52 → 9.03.**
`cd82` goes from 0.36 to **112.9**: all six levels on every seed (7 of 30 levels before),
in 147–155 actions a run against the humans' 171. Every other game scores exactly as in
Phase 27: the solver also recognised a canvas in `ft09`, where a clicked tile turns into
one colour as if painted, found nothing to plan and gave the level back at once, at no
measurable cost.

| `cd82` level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| human actions | 55 | 8 | 41 | 21 | 23 | 23 |
| solver actions (mean of 5 seeds) | 65.6 | 7.0 | 34.0 | 15.2 | 14.6 | 16.0 |
| level score | 70 | 115 | 115 | 115 | 115 | 115 |

The first level is the only one above the human count, and on purpose: about 50 of its
actions are the survey (eight cursor places, every move tried at each, every brush once).
It is what makes the second level cost 7 actions where a human needs 8: the second picture
needs a triangle brush the first never uses. The third level brings stamps (clicking the
small shape beside the cursor paints a block); the first stamp that paints sends the solver
round all eight places, and from then on every picture is planned outright.

Three things the first version got wrong, kept as lessons:

* **A use shows less than the brush.** Pixels that already had the colour do not change,
  so the region a brush is seen to paint is smaller than the region it paints, and plans
  built on it never converge. Completing the region with the shapes that fit, and painting
  in the colour the canvas has least of while surveying, fixed it.
* **The cursor must be read without the things that come and go next to it.** With the
  whole margin as the cursor state, the stamps that appear at the third level made every
  place look new, and nothing learned at the first two levels applied.
* **A status bar changes at every action**, so "the frame changed" is not evidence that a
  click did something; the bar is masked before asking.

The rest of the family is not covered, and the solver says why by staying silent: in
`tr87` the output tiles cycle through glyphs by rules shown as pairs, in `sk48` a chain is
extended and slid to pick up beads, in `sb26` a program frame is filled and run; none has a
canvas that one action paints in one colour. Each would need its own solver.

### Where the library stands

| | levels (4,000 steps, 5 seeds) | official-style score |
|---|---:|---:|
| model-guided agent alone (Phase 23) | 113 | 0.44 |
| library, marker family (Phase 25) | 143 | 4.52 |
| library, + navigation and hand-over (Phases 26–27) | 141 | 4.52 |
| library, + copy family (Phase 28, default) | 164 | **9.03** |

All of the gain is two families. The marker family's share of the mean went from 0.24 to
4.33, and its cost on the levels it completes from 21.8 to 6.3 times the human count; the
copy family's went from 0.01 to 4.52, at 0.9 times the human count (all of it `cd82`).
`lp85` is solved
outright (every level, every seed, in about half the human actions); `re86` and `s5i5`
open at human cost and stop at their second level; `vc33` and `r11l` keep what the
model-guided agent already did. Two lessons:

* **A family solver pays only where the family is the whole game.** `lp85` is nothing but
  rings, buttons and marks, and `cd82` nothing but a canvas, brushes and a palette, and the
  solvers finish both. In every other game something
  beyond the shared core decides the later levels — paint pots and benders (`re86`), arms
  that collide and carry children (`s5i5`), columns that run out and bridges that swap
  (`vc33`), handles that go wherever the player clicks (`r11l`), and in navigation each
  game's own spells, ghosts and glyphs.
* **Learning from the completed level is the strongest generic signal.** The first frame of
  a level-up still shows the finished level; the arrangement of piece and mark in it, and
  the controls that moved the pieces, carry over to the next level of the same game. It is
  what took `lp85` from five levels to eight.

The evaluation games are different games, so these numbers are an upper bound on what the
library transfers: it should help on a new game where pieces go onto marks of their colour
and be inert elsewhere (it takes control only with a plan in hand, gives back after a few
failures, and costs at most about 20 ms a step).

## The competition metric, and what it says about this agent

The levels-per-budget numbers above are the right measure for the "beyond" strand (does
the agent learn its world?) but not for the competition. ARC-AGI-3 scores a completed
level as `min(115, (human_actions / agent_actions)² × 100)`, where the human figure is the
median of first-time players, weights levels by their 1-indexed number, and averages
over games. Every action between two level completions counts, exploration included.
`results/summary.md` now reports this score too. On the public games:

| agent | levels (4,000 steps) | official-style score (0–100) |
|---|---|---|
| graph exploration | 17 / 75 | 0.26 |
| model, Phase 10 | 20 / 95 | 0.39 |
| model, Phase 11 | 21 / 98 | 0.42 |
| model, Phase 12 | 23 / 110 | 0.44 |
| model, Phase 13 | 23 / 110 | 0.44 |
| model, Phases 15–16 | 23 / 110 | 0.44 |
| model, Phases 17–18 | 23 / 110 | 0.44 |
| model, Phases 22–23 | 23 / 113 | 0.44 |
| library, Phase 25 (marker solver) | 143 levels | **4.52** |
| library, Phases 26–27 | 141 levels | 4.52 |
| library, Phase 28 (copy solver, default) | 164 levels | **9.03** |

The reported top of the Kaggle leaderboard is in the fifties. The gap is not about how
many levels are completed but about **how many actions each one costs**: `ls20`'s first
level takes this agent ~200–1,300 actions against a human's 22, which is worth 0–1 % of
that level; `r11l`'s first level in 41 actions against 22 is worth 29 %, its second in
897 against 33 is worth 0.1 %. Exploration is what the metric punishes, quadratically.

Two consequences shape what comes next. First, a level's weight grows with its index, so
a game's score is dominated by its later levels: an agent that spends the first level
learning and then solves later levels in near-human action counts can score well even
with an expensive first level — which is exactly where a learned world model and a
planner should pay. Second, nothing in a tabula-rasa explorer can make the *first* level
of a new game cheap; that needs priors about what games tend to be, which is the
language-model direction (phase 10 in the plan) or a library of mechanics learned
across games.

## Conclusion and next step

What moved the needle was Phase 2: rules induced from the agent's own transitions —
which object the keyboard moves and how far, which colours block it, which clicks never
do anything, which actions or cells lose the game — used to prune and order the
exploration. Everything after that (planning towards past goals, state abstraction,
curiosity, instance-level rules, object-targeted exploration) was measured and found
neutral or negative at 4,000 steps, and is kept only as an opt-in variant. The gain from
the rules grows with the budget: +47 % levels per run at 4,000 steps, +51 % at 24,000
(the latter measured before Phases 10–12). Phases 11 and 12 showed that a planner is
only as good as the world model under it: fixing how a jumping avatar's moves are
credited, and explaining a loss before blaming it, were worth more on `tu93` and `sp80`
than any planning heuristic tried before them.

The remaining losses are of one kind. What completes a level, and what loses the game,
is relational — *which* instance of an object, *where*, in *what* state, *pushed* against
what — and a model over colours and signatures cannot express it. Phase 8b took the first
step (goals as object signatures, +1 median level); Phase 8c (adjacency) was neutral.
Phases 11–12 fixed the world model under the planner (sweeps, death attribution) and were
worth +0.05 on the official score; Phases 13–14 were neutral and negative, and the loop
stopped there by its own rule.

**Where this stood after Phase 23 (model-guided agent, 4,000 steps, 5 seeds):** 23 median
levels, 110 levels over all runs, official-style score 0.44 against 0.26 for the
exploration baseline. The score was low in absolute terms because every public game's
first level is found by exploration at tens of times the human action count, and because
the games where later levels are reached cheaply (`tu93`, `sp80`, `vc33`, `r11l`, `lp85`)
are few.

**Where it stands now (the `library` agent, Phases 24–28):** 164 levels, official-style
score **9.03** — twenty times Phase 23, all of it from two family solvers: pieces onto the
mark of their colour (4.33 of the mean, 4.06 of it `lp85`) and a canvas painted like its reference
(`cd82`, 4.52). The reported top of the Kaggle leaderboard is in the fifties; the agent is
not ready to submit.

**What would move it next after the library** (Phases 24–28): the remaining points sit in
each game's own mechanism. The families whose goal is *shown* on screen pay best, because
the goal need not be discovered: Phase 28 took the canvas-against-reference case (`cd82`)
from 0.36 to 112.9. The other shown-goal games each need a solver of their own — an
output row against rewrite rules (`tr87`), a chain against a template (`sk48`), a painted
sequence against target boxes (`sb26`) — and the marker games stop at their second level
on mechanisms beyond the shared core (`re86`, `s5i5`). Before the library, the list read:

**What would move it next (as of Phase 23)**, in order of expected value:

1. *State preconditions on goals.* `ls20`'s box admits the avatar only when the pattern it
   carries matches; the pattern is cycled by floor tiles. The data to learn this is in the
   transitions already recorded (which sprite changed when the avatar touched what, and
   what the avatar looked like when a level ended); the model lacks a notion of avatar
   state. This is the general form of "collect the key, then the door".
2. *Relational hazards.* `tu93`'s patrol charges along its row or column when the avatar
   enters it; the reach model (one step around it) is too coarse to plan through, so the
   agent gets past by re-testing. A rule of the form "kind X moves towards the avatar when
   aligned" is checkable on the recorded frames.
3. *Tile mechanics.* `dc22`'s clicks slide blocks that the avatar then walks on. Clicking
   is modelled per target only; what a click does *to the map* is not.
4. *Priors.* Nothing above makes the first level of a new game cheap; that needs a library
   of mechanics learned across games or a language model proposing hypotheses for the data
   to check — the plan's optional phase.

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
