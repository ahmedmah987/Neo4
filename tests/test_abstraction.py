"""Noise detection and state abstraction, on a corridor with drifting particles."""

import random

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from neo4 import frame as fp
from neo4.graph_agent import GraphAgent
from neo4.model_agent import ModelAgent
from neo4.objects import erase, extract


class Snowfield:
    """Avatar moves 4 px right/left with ACTION1/2; ACTION3/4 do nothing.
    Eight 1x1 particles of colour 7 jump to random places every step.
    Reaching x >= 40 completes the level."""

    def __init__(self, seed=0, particles=8):
        self.rng = random.Random(seed)
        self.n = particles
        self.x, self.level, self.state = 4, 0, GameState.NOT_PLAYED

    def _obs(self):
        o = FrameDataRaw(state=self.state, levels_completed=self.level, win_levels=1,
                         available_actions=[0, 1, 2, 3, 4])
        g = np.zeros((fp.GRID, fp.GRID), dtype=np.int8)
        g[:, 60:64] = 12  # the goal strip
        g[30:34, self.x : self.x + 4] = 9
        for _ in range(self.n):
            g[self.rng.randrange(0, 28), self.rng.randrange(0, 64)] = 7
        o.frame = [g]
        return o

    def reset(self):
        self.x, self.state = 4, GameState.NOT_FINISHED
        return self._obs()

    def step(self, action, data=None):
        if action is GameAction.RESET:
            return self.reset()
        if action is GameAction.ACTION1 and self.x < 56:
            self.x += 4
        elif action is GameAction.ACTION2 and self.x > 0:
            self.x -= 4
        if self.x >= 56:
            self.level, self.state = 1, GameState.WIN
        return self._obs()


def play(agent, env, steps):
    obs = env.reset()
    for used in range(1, steps + 1):
        a, d = agent.act(obs)
        obs = env.step(a, d)
        if obs.state == GameState.WIN:
            return used
    return steps


def test_erase_paints_over_with_surrounding_colour():
    g = np.full((fp.GRID, fp.GRID), 3, dtype=np.int8)
    g[10, 10] = 7
    objs = extract(g)
    sig = next(o.sig for o in objs if o.color == 7)
    out = erase(g, objs, {sig})
    assert out[10, 10] == 3 and (out == 3).all()
    assert g[10, 10] == 7  # the input is untouched


def test_particles_are_classed_as_noise_and_states_collapse():
    model = ModelAgent(seed=1, abstract=True)
    steps_model = play(model, Snowfield(seed=1), 600)
    noise = model.dyn.noise()
    assert any(sig[0] == 7 for sig in noise), model.dyn.describe()
    assert all(sig[0] != 9 for sig in noise)
    assert model.dyn.avatar() is not None and model.dyn.avatar()[0] == 9
    assert model.stats["graph_rebuilds"] >= 1
    # Without abstraction every frame is a new state and the corridor is never finished.
    plain = GraphAgent(seed=1)
    steps_plain = play(plain, Snowfield(seed=1), 600)
    assert steps_model < steps_plain, (steps_model, steps_plain)
    assert len(model.graph.nodes) < 60
