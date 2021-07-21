import logging
from itertools import groupby

import gym
# noinspection PyUnresolvedReferences
import gym_maze
from lcs import Perception
from lcs.agents.macs.macs import Configuration, MACS

logging.basicConfig(level=logging.INFO)


def _calculate_knowledge(agent: MACS, env):
    transitions = env.env.transitions
    covered_transitions = 0

    debug = dict()
    debug[0] = set()
    debug[1] = set()
    debug[2] = set()

    for p0, a, p1 in transitions:
        p0p = Perception(list(map(str, p0)))
        p1p = Perception(list(map(str, p1)))
        anticipations = list(agent.get_anticipations(p0p, a))

        # accurate classifiers
        if len(anticipations) == 1 and anticipations[0] == p1p:
            debug[a].add(p0p)
            covered_transitions += 1

    return {
        'score': covered_transitions / len(transitions),
        'good_0_trans': len(debug[0]),
        'good_1_trans': len(debug[1]),
        'good_2_trans': len(debug[2]),
    }


def _metrics(agent: MACS, env):
    population = agent.population
    return {
        'pop': len(population),
        'situations': len(agent.desirability_values),
        '0_cls': len([cl for cl in population if cl.action == 0 and cl.is_accurate]),
        '1_cls': len([cl for cl in population if cl.action == 1 and cl.is_accurate]),
        '2_cls': len([cl for cl in population if cl.action == 2 and cl.is_accurate]),
        'knowledge': _calculate_knowledge(agent, env)
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

    agent = MACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 100)

    for action, gcl in groupby(sorted(pop, key=lambda c: (c.action, c.condition))):
        for cl in gcl:
            print(cl)
