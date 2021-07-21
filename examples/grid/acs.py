import logging

import gym  # noqa: E402
# noinspection PyUnresolvedReferences
import gym_grid  # noqa: E402
from lcs.agents.acs import ACS, Configuration

# Configure logger
from examples.grid import print_cl

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Load desired environment
    grid = gym.make('grid-10-v0')

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=2,
        number_of_possible_actions=4,
        epsilon=0.95,
        beta=0.01,
        theta_i=0.1,
        metrics_trial_frequency=10)

    # Explore the environment
    agent = ACS(cfg)
    population, explore_metrics = agent.explore(grid, 500)

    for cl in sorted(population, key=lambda c: -c.fitness):
        print_cl(cl)

    # Exploit
    agent2 = ACS(cfg, population)
    pop_exploit, metric_exploit = agent2.exploit(grid, 100)
