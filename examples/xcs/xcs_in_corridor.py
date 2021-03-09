import logging
import os
from lcs.agents.xcs import XCS, Configuration

import gym  # noqa: E402
# noinspection PyUnresolvedReferences
import gym_corridor  # noqa: E402


# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def print_cl(cl):
    print(str(cl))


if __name__ == '__main__':

    maze = gym.make('corridor-20-v0')

    cfg = Configuration(theta_mna=2)

    agent = XCS(cfg)

    population, explore_metrics = agent.explore(maze, 500)

    for cl in population:
        print_cl(cl)

    population, explore_metrics = agent.exploit(maze, 500)

    for cl in population:
        print_cl(cl)
