import gym
import numpy as np


class CartPoleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, bins, low, val_range):
        super().__init__(env)
        self.bins = bins
        self.low = low
        self.range = val_range

    def observation(self, observation):
        return np.round(((observation - self.low) / self.range) * self.bins) \
            .astype(int) \
            .astype(str) \
            .tolist()
