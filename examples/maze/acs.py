import argparse
import logging

import gym
# noinspection PyUnresolvedReferences
import gym_maze

from examples.maze.utils import maze_knowledge
from lcs.agents.acs import ACS, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def maze_metrics(agent, environment):
    return {
        'population': len(agent.population),
        'knowledge': maze_knowledge(agent.population, environment)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="MazeF1-v0")
    parser.add_argument("--epsilon", default=0.5, type=float)
    parser.add_argument("--explore-trials", default=1000, type=int)
    parser.add_argument("--exploit-trials", default=100, type=int)
    args = parser.parse_args()

    # Load desired environment
    maze = gym.make(args.environment)

    # Configure and create the agent
    cfg = Configuration(classifier_length=8,
                        number_of_possible_actions=8,
                        epsilon=args.epsilon,
                        metrics_trial_frequency=5,
                        user_metrics_collector_fcn=maze_metrics)

    # Explore the environment
    logging.info("Exploring maze")
    agent = ACS(cfg)
    population, explore_metrics = agent.explore(maze, args.explore_trials)

    for metric in explore_metrics:
        logger.info(metric)

    for cl in sorted(population, key=lambda c: -c.fitness):
        print(cl)

    # Exploit the environment
    logging.info("Exploiting maze")
    agent = ACS(cfg, population)
    population, exploit_metric = agent.exploit(maze, args.exploit_trials)

    for metric in exploit_metric:
        logger.info(metric)

    logger.info("Done")
