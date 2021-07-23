import gym
import numpy as np


class MountainCarObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, bins):
        super().__init__(env)
        self.bins = bins
        self._range = env.observation_space.high - env.observation_space.low
        self._low = env.observation_space.low
    
    def observation(self, obs):
        r = (obs + np.abs(self._low)) / self._range
        b = (r * self.bins).astype(int)
        return b.astype(str).tolist()
