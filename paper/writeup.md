# Search is not the bottleneck: a mechanics library for ARC-AGI-3, and what it fails to transfer

*Code (MIT-0): https://github.com/ahmedmah987/neo4.
Kaggle submission: ARC Prize 2026 – ARC-AGI-3, notebook `ahmedabdalatef/neo4-arc-agi-3`.*

## Abstract

We built a frames-only agent for ARC-AGI-3 in two layers. The first is a model-guided explorer:
it induces rules from its own transitions and uses them to prune, order and plan. The second is
a library of eighteen mechanic solvers. Each recognises its mechanic from the frames, models it
exactly and plans the fewest actions. On the 25 public games the library raises the
official-style score from 0.44 to 80.01, sixteen games at the cap. On the hidden games it
scores 0.33. We measure the gap offline with a holdout protocol, in which each public game is
played without the solvers built for it. That gives 0.45, which predicts the hidden score.
Sixteen public games were diagnosed by search on hand-built models of their rules, and every
level was won in no more actions than humans took. Search, then, is not the bottleneck:
acquiring the model is. We report what transfers, what does not, and a first step towards
learning the goal from completed levels.

## 1. Introduction

ARC-AGI-3 games are turn-based, on a 64×64 grid of 16 colours, with no instructions. A level
scores min(115, (h/a)²·100), where h is the human action count and a the agent's. Levels are
weighted by their index. Two consequences shaped this work. Exploration is punished
quadratically: a level finished in ten times the human count is worth 1%. And later levels,
where a learned rule can be reused, weigh the most.

We started from graph-based exploration [1] and asked how far an agent without a language
model can go. Over 57 measured phases we answer two questions: how well can such an agent
play once it understands a mechanic, and how much of that understanding transfers?

## 2. Prior work

The graph explorer of [1] tests every action in every state; we keep its state graph and add
induced rules. The leading Kaggle entries this year instead run local language or vision
models that write code or pick actions [2], and score 45–71 on the hidden games. Our agent
uses no learned model weights.

## 3. Approach

**Perception.** A frame is a set of single-colour components, each with a signature (colour,
size, shape), matched between frames; status bars and action-independent noise are masked.

**The model-guided explorer.** Every transition updates a rule set:
* which object the keys move, by how much, and which colours block it;
* which click targets never change anything;
* which actions or cells end the game, and move budgets and resource bars;
* switches that gate a goal.

Before trying a candidate action, the agent asks the model. Predicted no-ops and deaths are
skipped; unpredictable actions go first. After a level-up the agent plans towards what
completed it (the colour the avatar entered, the kind that was clicked) through cells known to
be passable. It also orders clicks by how much each kind closed a learned goal relation.

**The mechanics library.** Eighteen family solvers sit in front of the explorer, each a
perception step, an exact model and a planner. Examples:
* *maze*: an avatar to an exit past watchers whose rules are learned by watching;
* *keys*: a key turned, recoloured and reshaped on tiles until it matches a door, partly in fog;
* *gravity*: a shaft taller than the screen, with clicks that clear, turn, toggle or grow tiles;
* *mirror*: pieces and mirror lines moved until pieces and reflections cover every target;
* *pour*, *spells*, *panel*, *replay* (ghosts that re-walk a run), *program*, *translate*,
  *bit-program*, *clue board*, *connector*, *carry*, *meet*, *painter*, *pointer*, *marker*.

A solver claims a game only when its perception reads a consistent scene, which we check on
every public game for false positives. It plays only while the frames match its predictions,
then hands back to the explorer. Two design rules mattered most:
1. **Plan where things end.** When nothing collides, the search runs over final placements,
   not action sequences: `ar25` is solved in the fewest possible actions, in under three
   seconds a level.
2. **States hold only what can still matter.** Leaving out tiles sealed in a wall and toggles
   set just before use took one level's search from out of memory to 88 states.

**Diagnoses.** Before writing a solver we read each hard game's source and played every level,
by search on a hand-built model or by hand. This bounds what perfect understanding is worth.

**Holdout evaluation.** To estimate performance on unseen games offline, each public game is
played with the solvers built or tuned on it removed; everything else stays.

## 4. Results

Official-style score, 25 public games, 4,000 actions, 5 seeds:

| agent | levels | score |
|---|---:|---:|
| exploration baseline [1] | 75 | 0.26 |
| model-guided explorer | 113 | 0.44 |
| + mechanics library (eighteen solvers) | 653 | **80.01** |
| holdout: each game without its own solvers | — | 0.45 |
| **hidden games (Kaggle public leaderboard)** | — | **0.33** |

Sixteen of the 25 games are played at or near the cap. Examples: `ar25`, all eight levels in
253 actions (humans 748); `bp35`, all nine in 448 (651); `tu93` in the fewest moves its mazes
allow. Sixteen games were diagnosed while the library could not yet play them. Every level of
each was won in no more than the humans' actions; the last eight took 2,398 actions against
6,608.

The holdout score matches the explorer's and predicts the hidden score. The solvers neither
help nor hurt a game they were not built for. They claim such games briefly and give way,
and do not transfer between public games at all. Within a mechanic they do transfer: each of
the eighteen plays a toy game written apart from it, with other colours, sizes and layouts.
What they encode is a mechanic, and the public games share almost none.

## 5. Analysis: why the gap

Three findings explain the numbers.

1. **Understanding is worth everything, and search is cheap.** For every diagnosed game,
   search on the right model finds solutions within human action counts; our solvers' plans
   take seconds. Where the explorer lacks the model, its levels cost 10–50 times the human
   count and score nothing.
2. **The explorer's points come from first levels with few relevant actions.** It finishes
   some first levels near human cost (`vc33`, `r11l`, `lp85`, `sp80`). Second levels often
   cost hundreds to thousands of actions: `r11l` 645–1,934 against the humans' 33, and `sp80`
   429–2,402 against 58. What the agent learned about the goal does not survive the new
   layout.
3. **Goals, not mechanics, are what the explorer fails to learn.** Its rules capture movement
   and blocking well. Its goal is guessed from one level-up as a gap between two kinds, and a
   level-1 solution path is only 3–8 actions long, so many measures change together along it.
   Replaying the last solution transfers nothing; ordering clicks by a learned relation helps
   only where the guessed relation is the true one.

## 6. Towards closing it

The library contains the skills; what is missing is knowing which applies. As a first step,
goal hypotheses now narrow from level to level. Each completed level proposes candidates:
gaps between kinds, and counts of a kind or a colour that only ever moved towards where they
ended. Hypotheses an earlier level also supported rank first. Together they direct click
ordering and frontier choice on the next level. On the toggle game `ft09`, a colour count
gives a second level under every hash seed we tried and none without it, in 22 to 448
actions (humans 12). One holdout sweep
rose from 0.448 to 0.471, but sweeps vary by about ±0.02 with Python's hash order; under
fixed hash seeds the expected gain is +0.01. Skipping actions the explorer
can see are wasted, an undo key and keys that never change anything, adds +0.003. The
direction is right, but one short solution path leaves too many hypotheses standing. Beyond
that, we see two steps.
* Split each solver into perception and skill, so that a skill (navigate, push, select and
  move, toggle, cover) can be bound to roles learned by interaction.
* Measure transfer with holdouts during development, not after.

## 7. Conclusion

A frames-only agent with hand-built mechanic solvers plays the public ARC-AGI-3 games near
human efficiency, and the hidden ones barely at all. The holdout protocol predicts this
offline. The diagnoses show that search on a correct model is never the limit. The open
problem is learning, from a few levels, what a game's objects are, what its actions do and
what it asks, in a form a planner can use.

**References.** [1] *Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks*,
AAAI 2026 workshop, and `arc-agi-3-just-explore`. [2] ARC Prize 2026 Milestone #1 report,
arcprize.org.
