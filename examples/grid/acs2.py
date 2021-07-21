import logging

import gym  # noqa: E402
# noinspection PyUnresolvedReferences
import gym_grid  # noqa: E402
from lcs import Perception
from lcs.agents.acs2 import ACS2, Configuration
from lcs.metrics import population_metrics
from lcs.strategies.action_selection.BestAction import BestAction

from examples.grid import GridObservationWrapper, grid_transitions, print_cl

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTIONS = ['⬅', '➡', '⬆', '⬇']

# Define grid size
grid_size = 5


def calculate_knowledge(population, environment):
    reliable = [c for c in population if c.is_reliable()]
    nr_correct = 0

    mappings = {}

    for start, action, end in transitions:
        mappings[(start, action, end)] = []

    for start, action, end in transitions:
        p0 = Perception([str(el) for el in start])
        p1 = Perception([str(el) for el in end])

        if any([True for cl in reliable if
                cl.predicts_successfully(p0, action, p1)]):
            mappings[(start, action, end)] = [cl for cl in reliable if
                                              cl.predicts_successfully(p0,
                                                                       action,
                                                                       p1)]
            nr_correct += 1

    return nr_correct / len(transitions) * 100.0


def optimal_action_prob(population, environment):
    m = {}

    for state, _, _ in transitions:
        p0 = Perception([str(el) for el in state])
        match_set = population.form_match_set(p0)
        m[state] = BestAction(4)(match_set)

    return sum(1 for a in m.values() if a in [1, 2]) / len(transitions) * 100


transitions = grid_transitions(grid_size)


# Build agent configuration
# Collect additional population metrics
def grid_metrics(agent, env):
    pop = agent.population
    metrics = {
        'knowledge': calculate_knowledge(pop, env),
        'optimal_action_prob': optimal_action_prob(pop, env)
    }
    metrics.update(population_metrics(pop, env))
    return metrics


if __name__ == '__main__':
    # Load desired environment
    grid = GridObservationWrapper(gym.make(f'grid-{grid_size}-v0'))

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=2,
        number_of_possible_actions=4,
        epsilon=0.9,
        beta=0.2,
        gamma=0.95,
        theta_i=0.1,
        theta_as=50,
        theta_exp=50,
        theta_ga=50,
        do_ga=False,
        mu=0.04,
        u_max=2,
        metrics_trial_frequency=5,
        user_metrics_collector_fcn=grid_metrics)

    # Explore the environment
    agent1 = ACS2(cfg)
    population, explore_metrics = agent1.explore(grid, 500, decay=False)

    for cl in sorted(population, key=lambda c: -c.fitness):
        if cl.does_anticipate_change():
            print_cl(cl)

    # Exploit
    agent2 = ACS2(cfg, population)
    pop_exploit, metric_exploit = agent2.exploit(grid, 30)

    # Print classifiers
    for cl in sorted(pop_exploit, key=lambda c: -c.fitness):
        if cl.does_anticipate_change():
            print_cl(cl)
