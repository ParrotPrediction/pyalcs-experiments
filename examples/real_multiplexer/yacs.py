import logging

import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer
from lcs.agents.yacs.yacs import Configuration, YACS

from examples.real_multiplexer import HashedObservation

logging.basicConfig(level=logging.INFO)

RMPX_SIZE = 3
RMPX_HASH = 'md5'
RMPX_BINS = 10

if __name__ == '__main__':
    cfg = Configuration(classifier_length=RMPX_SIZE + 1,
                        number_of_possible_actions=2,
                        learning_rate=0.1,
                        discount_factor=0.8,
                        trace_length=3,
                        estimate_expected_improvements=False,
                        feature_possible_values=[{str(i) for i in range(RMPX_BINS)}] * RMPX_SIZE + [{'F', 'T'}],
                        metrics_trial_frequency=1)

    env = HashedObservation(
        gym.make(f'real-multiplexer-{RMPX_SIZE}bit-v0'), RMPX_HASH, RMPX_BINS)

    agent = YACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 500)
