import logging

import gym
# noinspection PyUnresolvedReferences
import gym_corridor
from lcs.agents.yacs.yacs import Configuration, YACS

from examples.corridor import CorridorObservationWrapper

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    env = CorridorObservationWrapper(gym.make('corridor-20-v0'))

    cfg = Configuration(classifier_length=1,
                        number_of_possible_actions=2,
                        learning_rate=0.1,
                        discount_factor=0.8,
                        trace_length=3,
                        estimate_expected_improvements=True,
                        feature_possible_values=[set(str(i) for i in range(19)),],
                        metrics_trial_frequency=1)

    agent = YACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 500)

    print("\n*** DESIRABILITY VALUES ***")
    for p, de in agent.desirability_values.items():
        state = env.env._state_id(list(map(int, p)))
        print(f"{state}:\t{de}")

    # _present_classifiers(pop, env)
    #
    # print("\n*** EXPLOIT ***")
    # cfg.metrics_trial_frequency = 1
    # exploiter = YACS(cfg, agent.population, agent.desirability_values)
    # exploiter.exploit(env, 1)
