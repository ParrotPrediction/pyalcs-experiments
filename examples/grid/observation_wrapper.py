import gym


class GridObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return observation
