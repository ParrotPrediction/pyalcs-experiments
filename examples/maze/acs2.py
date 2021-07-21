import argparse
import logging

import gym
# noinspection PyUnresolvedReferences
import gym_maze
from lcs.agents.acs2 import ACS2, Configuration

from examples.maze.utils import maze_knowledge

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def maze_metrics(agent, environment):
    population = agent.population
    return {
        'population': len(population),
        'knowledge': maze_knowledge(population, environment)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="Maze4-v0")
    parser.add_argument("--epsilon", default=0.5, type=float)
    parser.add_argument("--ga", action="store_true")
    parser.add_argument("--pee", action="store_true")
    parser.add_argument("--explore-trials", default=5000, type=int)
    parser.add_argument("--exploit-trials", default=10, type=int)
    args = parser.parse_args()

    # Load desired environment
    maze = gym.make(args.environment)

    # Configure and create the agent
    cfg = Configuration(classifier_length=8,
                        number_of_possible_actions=8,
                        epsilon=args.epsilon,
                        do_ga=args.ga,
                        do_pee=args.pee,
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=maze_metrics)

    # Explore the environment
    logging.info("Exploring maze")
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(maze, args.explore_trials)

    # TODO: check performance

    for metric in explore_metrics:
        logger.info(metric)

    # Exploit the environment
    logging.info("Exploiting maze")
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(maze, args.exploit_trials)

    for metric in exploit_metric:
        logger.info(metric)

    logger.info("Done")
