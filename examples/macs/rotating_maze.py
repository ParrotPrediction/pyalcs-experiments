import gym
# noinspection PyUnresolvedReferences
import gym_maze
import logging
from lcs.agents.macs.macs import Configuration, MACS

logging.basicConfig(level=logging.INFO)


def _metrics(agent, env):
    pop_len = len(agent.population)
    return {
        'pop': pop_len,
        'accurate': len([cl for cl in agent.population if cl.is_accurate]),
    }


if __name__ == '__main__':
    env = gym.make('Maze228-v0')

    state_values = {'0', '1', '9'}

    cfg = Configuration(classifier_length=8,
                        number_of_possible_actions=3,
                        feature_possible_values=[state_values] * 8,
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=_metrics)

    agent = MACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 100)

    for cl in pop:
        print(cl)
