from lcs.agents.xcs import XCS, Configuration
import gym
# noinspection PyUnresolvedReferences
import gym_corridor

import logging


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def xcs_corridor_knowledge(xcs: XCS, environment):
    nr_correct = 0
    position = 0
    while environment.observation_space.contains(position):
        if any([True for cl in xcs.population if cl.does_match(str(position))]):
            nr_correct += 1
        position += 1
    return nr_correct / position


def xcs_corridor_metrics(xcs: XCS, environment):
    return {
        'population': sum(cl.numerosity for cl in xcs.population),
        'knowledge': xcs_corridor_knowledge(xcs, environment)
    }


if __name__ == '__main__':

    maze = gym.make('corridor-20-v0')

    cfg = Configuration(number_of_actions=2,
                        user_metrics_collector_fcn=xcs_corridor_metrics)

    agent = XCS(cfg)

    # population, explore_metrics = agent.explore(maze, 500)

    population, explore_metrics = agent.exploit(maze, 500)

    # number of classifier in episode
    # number of steps towards solution in episode
    # some metric for knowledge or policy
    # (for now I am testing maze_knowledge from ACS2)

    for cl in population:
        print(str(cl))

    # {'trial': xxx, 'steps_in_trial': x, 'reward': xxxx, 'population': xx}
    for e in explore_metrics:
        print(e)
        # print(e["steps_in_trial"])
        # print(e["population"])

