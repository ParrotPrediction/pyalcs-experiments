import itertools
import numpy as np
import bitstring


CTRL_BITS = {
    3: 1,
    6: 2,
    11: 3,
    # ...
}


class RealMultiplexerUtils:
    def __init__(self, size, bins, env, _threshold=0.5):
        self._size = size
        self._ctrl_bits = CTRL_BITS[size]
        self._bins = bins
        self._threshold = _threshold

        self._range, self._low = (env.observation_space.high - env.observation_space.low, env.observation_space.low)
        self._step = self._range / self._bins

        self._attribute_values = [list(range(0, bins))] * (size) + [[0, bins]]
        self._input_space = itertools.product(*self._attribute_values)
        self.state_mapping = {idx: s for idx, s in enumerate(self._input_space)}
        self.state_mapping_inv = {v: k for k, v in self.state_mapping.items()}

    def discretize(self, obs, _type=int):
        r = (obs + np.abs(self._low)) / self._range
        b = (r * self._bins).astype(int)
        return b.astype(_type).tolist()

    def reverse_discretize(self, discretized):
        return discretized * self._step[:len(discretized)]

    def get_transitions(self):
        transitions = []

        initial_dstates = [list(range(0, self._bins))] * (self._size)
        for d_state in itertools.product(*initial_dstates):
            correct_answer = self._get_correct_answer(d_state)

            if correct_answer == 0:
                transitions.append(
                    (d_state + (0,), 0, d_state + (self._bins,)))
                transitions.append((d_state + (0,), 1, d_state + (0,)))
            else:
                transitions.append((d_state + (0,), 0, d_state + (0,)))
                transitions.append(
                    (d_state + (0,), 1, d_state + (self._bins,)))

        return transitions

    def _get_correct_answer(self, discretized):
        estimated_obs = self.reverse_discretize(discretized)
        # B = np.where(estimated_obs > self._threshold, 1, 0)
        bits = bitstring.BitArray(estimated_obs > self._threshold)
        _ctrl_bits = bits[:self._ctrl_bits]
        _data_bits = bits[self._ctrl_bits:]

        return int(_data_bits[_ctrl_bits.uint])
