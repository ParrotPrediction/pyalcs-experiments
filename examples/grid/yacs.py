import logging

import gym
# noinspection PyUnresolvedReferences
import gym_grid
from lcs.agents.yacs.yacs import Configuration, YACS

from examples.grid import GridObservationWrapper

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    env = GridObservationWrapper(gym.make('grid-20-v0'))

    cfg = Configuration(classifier_length=2,
                        number_of_possible_actions=4,
                        learning_rate=0.1,
                        discount_factor=0.8,
                        trace_length=3,
                        estimate_expected_improvements=True,
                        feature_possible_values=[
                            set(str(i) for i in range(20)),
                            set(str(i) for i in range(20))
                        ],
                        metrics_trial_frequency=1)

    agent = YACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 50)

    print("\n*** DESIRABILITY VALUES ***")
    for p, de in agent.desirability_values.items():
        state = env.env._state_id(list(map(int, p)))
        print(f"{state}:\t{de}")
