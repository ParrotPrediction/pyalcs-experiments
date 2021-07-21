import gym


class TaxiObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return str(observation),
