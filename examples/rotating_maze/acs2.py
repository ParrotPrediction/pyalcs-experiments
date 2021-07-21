import logging

import gym
# noinspection PyUnresolvedReferences
import gym_maze
from lcs import Perception
from lcs.agents.acs2 import Configuration, ACS2

logging.basicConfig(level=logging.INFO)


def _calculate_knowledge(reliable, env):
    transitions = env.env.transitions
    covered_transitions = 0

    for p0, a, p1 in transitions:
        p0s = Perception(list(map(str, p0)))
        p1s = Perception(list(map(str, p1)))
        if any([cl.predicts_successfully(p0s, a, p1s) for cl in reliable]):
            covered_transitions += 1

    return covered_transitions / len(transitions)


def _metrics(agent: ACS2, env):
    population = agent.population
    reliable = [cl for cl in population if cl.is_reliable()]

    return {
        'pop': len(population),
        'rel': len(reliable),
        'knowledge': _calculate_knowledge(reliable, env)
    }


if __name__ == '__main__':
    env = gym.make('Maze228-v0')

    cfg = Configuration(classifier_length=9,
                        number_of_possible_actions=3,
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=_metrics,
                        do_ga=False)

    agent = ACS2(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 5000)

    for cl in pop:
        print(cl)
