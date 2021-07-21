import gym


class CorridorObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return observation,
