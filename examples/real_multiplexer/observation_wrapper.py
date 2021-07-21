import hashlib

import gym


class HashedObservation(gym.ObservationWrapper):

    def __init__(self, env, hash_name, modulo):
        super().__init__(env)
        self.hash_name = hash_name
        self.modulo = modulo

    def observation(self, obs):
        hashed = []

        # hash all attributes except last one
        for i in [str(i).encode('utf-8') for i in obs[:-1]]:
            h = hashlib.new(self.hash_name)
            h.update(str(i).encode('utf-8'))
            hash = int(h.hexdigest(), 16)

            hashed.append(str(hash % self.modulo))

        if obs[-1] == 0.0:
            hashed.append('F')  # false anticipation
        else:
            hashed.append('T')  # true anticipation

        return hashed
