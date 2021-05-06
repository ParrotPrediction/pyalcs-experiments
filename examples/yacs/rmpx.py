import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer
import logging

import bitstring
import itertools
import numpy as np

from lcs.agents import EnvironmentAdapter
from lcs.agents.yacs.yacs import Configuration, YACS

logging.basicConfig(level=logging.INFO)


RMPX_SIZE = 3
RMPX_BINS = 4

env = gym.make(f'real-multiplexer-{RMPX_SIZE}bit-v0')
_range, _low = (env.observation_space.high - env.observation_space.low, env.observation_space.low)


class RealMultiplexerUtils:
    def __init__(self, size, ctrl_bits, bins, _range, _threshold=0.5):
        self._size = size
        self._ctrl_bits = ctrl_bits
        self._bins = bins
        self._step = _range / bins
        self._threshold = _threshold

        self._attribute_values = [list(range(0, bins))] * (size) + [[0, bins]]
        self._input_space = itertools.product(*self._attribute_values)
        self.state_mapping = {idx: s for idx, s in enumerate(self._input_space)}

    def discretize(self, obs, _type=int):
        r = (obs + np.abs(_low)) / _range
        b = (r * RMPX_BINS).astype(int)
        return b.astype(_type).tolist()

    def reverse_discretize(self, discretized):
        return discretized * self._step[:len(discretized)]

    def get_transitions(self):
        transitions = []

        initial_dstates = [list(range(0, self._bins))] * (self._size)
        for d_state in itertools.product(*initial_dstates):
            correct_answer = self._get_correct_answer(d_state)

            if correct_answer == 0:
                transitions.append((d_state + (0,), 0, d_state + (self._bins,)))
                transitions.append((d_state + (0,), 1, d_state + (0,)))
            else:
                transitions.append((d_state + (0,), 0, d_state + (0,)))
                transitions.append((d_state + (0,), 1, d_state + (self._bins,)))

        return transitions

    def _get_correct_answer(self, discretized):
        estimated_obs = self.reverse_discretize(discretized)
        # B = np.where(estimated_obs > self._threshold, 1, 0)
        bits = bitstring.BitArray(estimated_obs > self._threshold)
        _ctrl_bits = bits[:self._ctrl_bits]
        _data_bits = bits[self._ctrl_bits:]

        return int(_data_bits[_ctrl_bits.uint])


rmpx_utils = RealMultiplexerUtils(RMPX_SIZE, 1, RMPX_BINS, _range)
a = rmpx_utils.get_transitions()


class RealMultiplxerAdapter(EnvironmentAdapter):
    @classmethod
    def to_genotype(cls, obs):
        return rmpx_utils.discretize(obs, _type=str)


def rmpx_transitions(rmpx_size, rmpx_bins):
    transitions = []

    initial_dstates = [list(range(0, rmpx_bins))] * (rmpx_size)
    for d_state in itertools.product(*initial_dstates):
        transitions.append((d_state + (0,), 0, d_state + (0,)))
        transitions.append((d_state + (0,), 1, d_state + (RMPX_BINS,)))

    return transitions


trans = rmpx_transitions(RMPX_SIZE, RMPX_BINS)

if __name__ == '__main__':
    cfg = Configuration(RMPX_SIZE+1, 2,
                        learning_rate=0.1,
                        discount_factor=0.8,
                        environment_adapter=RealMultiplxerAdapter,
                        trace_length=3,
                        estimate_expected_improvements=True,
                        feature_possible_values=[RMPX_BINS]*RMPX_SIZE + [2],
                        metrics_trial_frequency=1)

    agent = YACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 500)

