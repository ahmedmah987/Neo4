# Search is not the bottleneck: what a frames-only ARC-AGI-3 agent pays for, and why

*Code (MIT-0): https://github.com/ahmedmah987/neo4.
Kaggle submission: ARC Prize 2026 – ARC-AGI-3, notebook `ahmedabdalatef/neo4-arc-agi-3`.*

## Abstract

We built an ARC-AGI-3 agent with no language model and no learned weights: an explorer that
induces rules from its transitions, eighteen exact mechanic solvers, and a layer that guesses
goals and tests them. The solvers score 80.01 on the 25 public games; without the solvers built
for each game, 0.45, close to the hidden score of 0.33. Search on hand-built models is never
the limit; the goal is. Guesses tested one at a time raise the hidden score to 0.46. Goals
whose conditions are learned from the level's own record complete more levels (114 → 122),
but not cheaply. Logging every life shows why: the winning life plays a later level at a
person's cost, and 95% of the actions go to lives that die learning the way. When a lost life
costs about a person's actions, a level learned in *L* lives is worth about 1/*L*² of its share.

## 1. Introduction

ARC-AGI-3 games are turn-based on a 64×64 grid of 16 colours, with no instructions. A level
scores min(1.15, *h*/*a*)², with *h* the median human's actions and *a* the agent's, counted
from the previous level-up with resets and lost lives included. Later levels weigh more.
Starting from graph-based exploration [1], we asked how far an agent without a language model
can go, in 70 phases measured on an offline holdout.

## 2. Related work

The leading Kaggle entries run open-weight language models in code-writing harnesses [2, 3]
(top ten: 47–76). Frontier-model agents write executable world models and plan in them.
Twin [4] finds that "inferring the right goal is the harder problem", and excludes a goal
reached without a level boundary. Tycho [5] asks when to build, repair or
bypass a model. Kepler [6] validates hypotheses as executable models and reaches 100 on the
public games. Earlier, EMPA [7] inferred game theories with their win conditions, and LIVE [8]
refined rules using the situations where they failed. We use a fixed vocabulary of integer goal
measures, so every step is deterministic and its effect isolable.

## 3. Approach

**Explorer.** Frames are parsed into single-colour components matched between frames. Each
transition updates rules: what the keys move and what blocks it, which clicks do nothing, what
ends the game, move budgets, resource bars, switches. Actions predicted useless or fatal are
skipped.

**Mechanics library.** Eighteen solvers (maze, keys, mirror, gravity, pour, spells, panel,
replay, program, translate, and others) each pair a perception step with an exact model and a
planner, and play only while the frames match their predictions. Before writing one, we played
every level of the game by search on a hand-built model of its rules.

**Holdout.** Each public game is played without the solvers built or tuned on it (5 seeds,
4,000 actions, fixed hash seeds).

**Goals as measures.** From a level's first frame the agent proposes goals: reach a kind seen
once, clear a repeated kind, make two panels match, give a shape one colour. Each is an integer
distance that can be steered by. With no goal known, the agent tests one guess at a time, from
the 10th action where an avatar moves and the 60th elsewhere. A guess met while the level goes
on is set aside. At a level-up, a guess that went from its start to its goal is confirmed and
steers later levels.

**Generated guesses.** A kind that moves on some actions and not others gives *bring* guesses
(to its colour, its own kind, or a kind seen once). A shape that changes colour gives *paint*
guesses, and a kind that vanishes gives *clear*.

**The record judges goals.** For each guess the agent keeps the times it was met while the
level went on, the guesses that life had met by then, and the actions taken while it was met.
At a level-up:

* a guess that explains the end and was never met before is confirmed alone;
* a guess met before is confirmed with a condition: another guess met in the winning life and
  at none of the earlier meetings (*given*: the key, then the door), or else the winning action,
  never taken while it was met (*then*: stand on the exit, then press);
* a guess with neither is refuted.

Later levels steer to the condition first. This keeps conditional goals that Twin's rule
drops, and applies LIVE's refinement to goals.

## 4. Results

| agent | holdout | hidden |
|---|---:|---:|
| explorer + mechanics library | 0.45 | 0.33 |
| + goals read from the screen | 0.53 | 0.36 |
| + tested one at a time | 0.62 | **0.46** |
| + generated guesses | 0.62 | 0.33 |
| + the record judges goals (without generated guesses) | 0.62 | *v9: pending* |

On the public games the exploration baseline [1] scores 0.26, the explorer 0.44 and the library
80.01, sixteen games at or near the cap. In the sixteen diagnosed games, search on the right
model won every level within the humans' action count.

## 5. Analysis

**The goal is the bottleneck.** Over 111 holdout levels, the winning life took 1.3 times the
humans' actions, all lives 16.6 times: 92% is learning.

**What did not help.** Local language models as advisors lowered the holdout (Gemma 3 4B by 6%,
Qwen2.5 3B by 42%, Gemma 3 12B by 20%). A 181 GB Qwen3.8-Flash-Next runs on the competition's
RTX Pro 6000 offline (its 95 GiB embedding table read from disk), but answers in 61–81 s with
goals like the 12B model's. Goal statistics shared across games ordered tests no better than
distance (0.59–0.62).

**Generated guesses find goals the list lacks, but late.** `cn04` and `ka59` completed first
levels without their own solvers. But a level found at 300 actions where a person needs 60 is
worth 4% of its share, and tested earlier they crowd out cheap exploration (0.59). Their hidden
score fell (0.46 → 0.33); five identical submissions of one public notebook ranged ±14% [9], so
part of that drop may be noise.

**The record.** 81% of the guesses that explained a level-up had been met earlier in the same
level, yet were confirmed. With the record, of 237 explanations, 49 were kept alone, 60 with a
*given* condition and 38 with a *then* action; 90 were refuted. Levels completed rose from 114
to 122 (121 under a second hash seed): `ar25`'s second level in five seeds instead of one, at
1,100–1,600 actions instead of 3,154; `r11l`'s for the first time. The score stayed at 0.620.

## 6. Where the actions go: lives

| levels completed | winning life ÷ humans | actions in earlier lives |
|---|---:|---:|
| first levels | 1.5× | 89% |
| later levels | 1.0× | 95% |

The 43 later levels cost 796 deaths. Traced one at a time, most lives end when a per-life move
budget runs out:

| game, level | lives end at | winning life |
|---|---|---:|
| `r11l` 2 | 60 (9 of 20 deaths) | 46 |
| `lp85` 2 | 60 (4 of 5) | 51 |
| `su15` 1 | 32–37 | 20 |
| `tu93` 3 | 35 (5 of 14) | 23 |
| `ar25` 2 | about 90 | 95 |

Once the agent knows the way it plays like a person; it pays for learning the way by dying.

**A bound.** Let a level take *L* lives. Say the winning life costs about *h*, and each lost
life costs *bh*. Then *a* ≈ *h*(1 + (*L*−1)*b*), and the level is worth
(1 + (*L*−1)*b*)⁻² of its share, or 1/*L*² when *b* = 1. Half the share needs
(*L*−1)*b* ≤ 0.41: less than half a person's count spent in lost lives. For example, `r11l`'s
second level (*h* = 33, a 60-action budget, 20–70 lost lives) costs 957–3,341 actions: at most
0.12% of its share, though its winning life is near human.

So carrying more between levels left the score in place. The metric rewards working out the
way before walking it: in a model, as Twin, Tycho and Kepler do with simulators written by
frontier models. Our partial models did not change the score: walks planned with walls and
patrols (0.624), clicks planned with a per-shape model of their effects (0.620), which lacks
two-step effects (select, then place).

## 7. Conclusion

A frames-only agent with hand-built solvers plays the public games near human efficiency and
the hidden ones barely, as the holdout predicts. Each bottleneck removed revealed the next:
search (never the limit), the goal (tested guesses: 0.33 → 0.46), what the goal requires (the
record: eight more levels), and the lives spent learning the way, punished as 1/*L*².

The next step without a language model is a simulator built from the rules the agent already
induces, in which to spend lives instead of in the game.

**References.** [1] Graph-based exploration for ARC-AGI-3, arXiv 2512.24156. [2] Tufa Labs,
Duck harness. [3] D. Franzen, ARC-AGI-3 Milestone 2 solution. [4] Twin, arXiv 2608.14490.
[5] Tycho, arXiv 2607.28287. [6] Kepler, arXiv 2610.00834. [7] Tsividis et al., EMPA, arXiv
2107.12544. [8] Shen & Simon, IJCAI 1989. [9] leoprovorov, "100 ideas", Kaggle notebook.
