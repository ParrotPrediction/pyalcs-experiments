import logging

import gym
# noinspection PyUnresolvedReferences
import gym_maze
from lcs.agents.yacs.yacs import Configuration, YACS

logging.basicConfig(level=logging.INFO)

def _metrics(agent: YACS, env):
    population = agent.population
    return {
        'pop': len(population),
        'situations': len(agent.desirability_values),
    }


if __name__ == '__main__':
    env = gym.make('Maze228-v0')

    state_values = {'0', '1', '9'}

    cfg = Configuration(classifier_length=9,
                        number_of_possible_actions=3,
                        feature_possible_values=[state_values] * 8 + [{'0', '9'}],
                        estimate_expected_improvements=True,
                        metrics_trial_frequency=10,
                        user_metrics_collector_fcn=_metrics)

    agent = YACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 100)

    for cl in pop:
        print(cl)
