import gym
# noinspection PyUnresolvedReferences
import gym_maze
from lcs.agents.xcs import XCS, Configuration, Classifier
from lcs import Perception

import logging
# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def predicts_successfully(cl: Classifier, perception: Perception, action):
    if cl.does_match(perception):
        if cl.action == action:
            return True
    return False


# same methodology as maze_knowledge in
# examples/acs2/maze/utils.py
# I needed to simplify few things that were not present in XCS
def xcs_maze_knowledge(population, environment) -> float:
    transitions = environment.env.get_all_possible_transitions()
    nr_correct = 0

    for start, action, end in transitions:
        perception = environment.env.maze.perception(*start)
        if any([True for cl in population
                if predicts_successfully(cl, perception, action)]):
            nr_correct += 1
    return nr_correct / len(transitions)


def xcs_maze_metrics(xcs: XCS, environment):
    return {
        'population': xcs.population.numerosity(),
        'knowledge': xcs_maze_knowledge(xcs.population, environment)
    }


if __name__ == '__main__':

    maze = gym.make('Maze4-v0')

    cfg = Configuration(number_of_actions=8,
                        user_metrics_collector_fcn=xcs_maze_metrics)

    logging.info("Exploring maze")
    agent = XCS(cfg)
    population, explore_metric = agent.explore(maze, 500)

    logging.info("Exploiting maze")
    agent = XCS(cfg, population)
    population, exploit_metric = agent.exploit(maze, 100)

    for cl in population:
        logger.info(str(cl))

    for metric in exploit_metric:
        logger.info(metric)
