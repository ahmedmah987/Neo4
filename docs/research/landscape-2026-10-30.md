# The ARC-AGI-3 landscape on 30 October 2026

Research notes taken three days before the Kaggle deadline (2 November), to place Neo4 among the other
entries and the published work. Nothing here was run; sources are listed at the end.

## Where Neo4 stands

| | public leaderboard score |
|---|---:|
| 1st (Yi-Chia Chen @ Fields Model) | 75.56 |
| 2nd (Tufa Labs) | 70.02 |
| 3rd–10th | 58.59 … 46.82 |
| 20th | 39.52 |
| Neo4 v7 (Phases 61+63) | 0.46 |
| Neo4 v8 (+ Phase 69) | 0.33 |

The scale is the same: the leaderboard reports RHAE in percent. About 4,500 teams have entered. In June
the first milestone was won with 1.21; the top score passed 50 in early October.

## What the leading entries do

All of them run an open-weight language model on the competition's single RTX Pro 6000, offline.

**The Duck harness** (Tufa Labs; Milestone 1 winner, open source). The model plays through one Python
tool: the game state is exposed as Python variables (an ASCII grid, a connected-component
segmentation with containment and adjacency, the history of transitions, the last action's result),
and the model writes code to inspect the board, test a mechanic, and call `action(...)`. Its
thinking, tool calls and results form one growing conversation; when the context is full, the
oldest turns are evicted. Tufa reported that hand-built tools hurt and that gains came from images
and better base models.

**Daniel Franzen's Milestone 2 solution** (open source, 27.89 on the leaderboard; the base of many
later entries). The Duck loop, with:

* model: Qwen3.8-Flash-Next, Intel's W4A16 AutoRound quantization (the model probed in Phase 68),
  its ~95 GiB per-layer embedding table kept in host RAM, an MTP draft head for speculative decoding,
  served by an SGLang fork tuned for the RTX Pro 6000; 10 parallel game streams, 128 Ki context,
  ~610 generated tokens a second overall, 93% of prompt tokens served from the prefix cache;
* perception: board images at 10× (from 4×; the largest single gain among the image changes), a
  difference image after every action, the death frame after a game over, and the intermediate
  animation frames exposed as Python variables (the strongest perception gain; `tn36` is
  "essentially unsolvable" without them);
* action: UNDO exposed; guards that stop a batch after an action that changes nothing inside the
  board, and stop actions after a game over or level end;
* memory: model-written Python functions kept across calls for the whole game; the structured
  text world model of the original harness switched off (the conversation itself kept enough);
* scheduling: all games started at once, ten admitted at a time by a priority that estimates the
  score still to win on the current level, the levels after it, and how much the level has already
  absorbed without completing.

**Milestone 1 runners-up** (Reki; "forge") used Gemma-4-31B as a vision-language policy returning
1–4 JSON actions per turn, with a reflection memory and click heuristics (small, rare-coloured,
button-like shapes first; object kinds whose clicks never change anything are dropped).

## What the leaderboard's noise and score breakdown say

From a public "book of ideas" of 145 experiments on Franzen's baseline (leoprovorov):

* five unchanged submissions of the same notebook scored 23.85 to 31.73: one submission cannot rank
  two versions within several points of each other;
* of the score a run could have earned, 46.5% was lost in levels it never reached, 10.5% in the level
  where it stopped, and only 2.5% in extra actions on levels it cleared;
* the chance that another 10,000 tokens on a level clears it falls from 0.26 to about zero; 93% of
  clears happen in a level's first 70,000 tokens.

For Neo4 this means the v7 (0.46) against v8 (0.33) difference may be within the noise of a single
submission, and that the score is decided by how many levels are reached at all.

## Published research (mostly frontier models, public set)

* **Executable world models.** A coding agent writes a Python model of the game, checks it against
  past transitions, and plans through it (arXiv 2605.05138: 58.1 RHAE with GPT-5.5). **Tycho**
  (2607.28287) frames a game as a deterministic Moore machine and compares policies: no world model
  79.1 RHAE, the agent editing a model 85.4, a builder sub-agent on request 88.5, automatic repair
  after every failed check 83.1 (Opus 4.8, same budgets). Its conclusion: accurate transitions are
  not enough; correct outcome (goal) prediction and deciding *when* to build, repair, use or bypass
  a model matter. **Kepler** (2610.00834) turns hypotheses into executable world models validated by
  retrospective transition checks and conditional prediction checks: 100.00 RHAE on the 25 public
  games with Claude Opus 5, at most the median human's actions on 181 of 183 levels, $778 of tokens.
* **Harnesses.** NVIDIA's AVO and the earlier VISTA reach 100 on the public set with Claude Opus 5,
  which scores about 30 without a harness. The ARC-AGI-3 paper reports Opus 4.6 at 0.0 without a
  harness and 97.1 with one on one game, and 0.0 under both on another.
* **Before language models took over:** the 2025 preview was won by StochasticGoose (Tufa Labs,
  12.58%), a learned model of which actions change the frame plus informed search; graph-based
  exploration (arXiv 2512.24156) is the basis of Neo4's first agent.

## What this means for Neo4

1. Neo4's thesis — the bottleneck is finding the goal, and guesses must be generated from what
   actions do and tested — is the thesis of the executable-world-model papers, which state the
   open question as Phase 69 found it: which hypothesis to build or test, and when.
2. On Kaggle the gap is not in that idea but in the guesser: a 30B-class model writing and testing
   code reaches 25–75, a fixed list of symbolic guesses 0.5.
3. Neo4's measured components that the leading harnesses also found useful: a frame difference by
   object (moved, recoloured, appeared, vanished), avoiding actions that change nothing, keeping
   what a level taught for the next one, UNDO.

## Rules that bear on any next step

* Offline evaluation on one RTX Pro 6000; the card counts against the same 30 weekly GPU hours.
* All code and methods must be open-sourced to be eligible for prizes.
* Submissions close 2 November; the paper is due 8 November. The Paper Prize scores accuracy (from
  the Kaggle score), universality, progress, theory, completeness and novelty (each 0–5), and must
  link a working Kaggle submission.

## Sources

* Kaggle leaderboard and public notebooks of `arc-prize-2026-arc-agi-3`, read through the Kaggle API on
  30 October 2026: `dfranzen/arc-agi-3-milestone-2-solution`,
  `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`, `leoprovorov/wanna-don-t-do-this-100-ideas`.
* D. Franzen, ARC-AGI-3 Milestone 2 write-up: https://github.com/da-fr/arc-agi-3-solution (WRITEUP.md)
* Tufa Labs, The Duck: https://github.com/Tufalabs/duck-harness
* ARC Prize, Milestone #1: https://arcprize.org/blog/arc-prize-2026-milestone-1
* ARC Prize 2026, ARC-AGI-3 competition and Paper Prize: https://arcprize.org/competitions/2026/arc-agi-3,
  https://arcprize.org/competitions/2026/paper
* ARC-AGI-3 paper: https://arxiv.org/abs/2603.24621
* Kepler: https://arxiv.org/abs/2610.00834 · Tycho: https://arxiv.org/abs/2607.28287 ·
  Executable World Models: https://arxiv.org/abs/2605.05138 · Graph-based exploration:
  https://arxiv.org/abs/2512.24156
* NVIDIA AVO: https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/

## Prior art for Neo4's own ideas (31 October)

| Neo4's idea (phase) | Published before | What is still Neo4's |
|---|---|---|
| The goal, not the dynamics, is the bottleneck (Phases 56–68: 92% of actions before the goal is known, 1.3× a person's actions once it is) | **Twin** (arXiv 2608.14490, Aug 2026, GPT-5.6 Sol): "building a usable world model is simpler than expected; inferring the right goal is the harder problem"; its unfinished games show "accurate dynamics, unresolved goals" — `sc25`, `sp80`, `tn36`, `ka59`, the games Neo4 also fails | the measurement for an agent with no language model, by level and life |
| Goal guesses as measures, tested one at a time; a guess met without a level end is dropped, one met at a level end confirmed (Phase 61) | **Twin**: a candidate goal predicate is planned to; "reached exactly with no level boundary" it is permanently excluded; a level boundary confirms it. **EMPA** (Tsividis et al., arXiv 2107.12544, 2021): Bayesian inference over game theories including win conditions | guesses as integer distances (a gradient to steer by), not yes/no predicates; the hidden-score effect (0.36 → 0.46) |
| Guesses generated from what an action changed (Phase 69) | **Twin**: candidates ranked by progress signals `color_gone`, `color_new`, `local_burst`, `big_change`, `frontier`, found by search inside its model. **LIVE** (Shen & Simon, IJCAI 1989): rules created and refined from surprises, where predictions fail | generated from observed events, not from search in a model; measured cost of testing them early (crowds out cheap exploration) |
| Which guess to test, and when | **Tycho** (arXiv 2607.28287) states it as the open question: when to build, repair, use or bypass a model | the measurements of Phase 69 (15.19 → 15.52 by when they are tested; 14.85 when the cheap ones go early) |
| Graph exploration of the level's states (Phase 1) | **Graph-based exploration** (arXiv 2512.24156) | none (it is the base) |
| A model's guess at the goal from the first frame (Phases 58, 68) | the Milestone 1 runners-up (Gemma-4-31B policies); the harness papers | the measurement that a 181 GB model's first-frame goals read like a 12B model's |

Ideas from this prior art that fit Neo4 without a language model:

* **Eliminate on the record** (Twin): a guess whose goal was met at any logged frame of the level
  without the level ending is false, without spending a test on it.
* **Score guesses by evidence, test by expected cost** (EMPA; Bayesian experimental design): rank
  tests by how likely a guess is given every frame seen, over the actions it would cost.
* **Tell surprises from noise** (Collins & Shen, 2017): a distinguishing experiment before a change
  is taken as evidence — the problem of patrols and animations in Phase 69.
* **Candidates from search in the agent's own model** (Twin): propose as goals the reachable states
  with salient changes (a colour gone, a big change), not only what has already happened.
* AERA (arXiv 2605.25931) argues the 25 public games are a weak test (most fall to trivial
  strategies) and the 55-game private set is the real one — a caution for any holdout proxy.

Sources: https://arxiv.org/abs/2608.14490 · https://arxiv.org/abs/2107.12544 ·
https://mlanthology.org/ijcai/1989/shen1989ijcai-rule · https://robots.isi.edu/SBL/ ·
https://arxiv.org/abs/2605.25931 · https://arxiv.org/abs/2510.19788 (WorldTest / AutumnBench)
