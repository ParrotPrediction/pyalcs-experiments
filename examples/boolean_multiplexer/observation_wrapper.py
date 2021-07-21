import gym


class MpxObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return [str(x) for x in observation]
