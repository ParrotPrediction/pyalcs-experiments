import logging

import gym
# noinspection PyUnresolvedReferences
import gym_corridor

from lcs.agents.acs2 import ACS2, Configuration

from lcs.strategies.action_selection import EpsilonGreedy

# Configure logger
from examples.corridor import CorridorObservationWrapper

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Load desired environment
    corridor = CorridorObservationWrapper(gym.make('corridor-40-v0'))

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=1,
        number_of_possible_actions=2,
        action_selector=EpsilonGreedy,
        epsilon=0.8,
        beta=0.03,
        gamma=0.97,
        theta_exp=50,
        theta_ga=50,
        do_ga=True,
        mu=0.02,
        u_max=1,
        metrics_trial_frequency=20)

    # Explore the environment
    logging.info("Exploring environment")
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(corridor, 1000)

    population = sorted(population, key=lambda cl: -cl.fitness)

    print("ok")

    for cl in population:
        print(cl)
