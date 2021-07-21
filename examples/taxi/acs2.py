import logging

import gym
# noinspection PyUnresolvedReferences
import gym_taxi_goal
from lcs.agents.acs2 import ACS2, Configuration

from examples.taxi import TaxiObservationWrapper, taxi_metrics

# Configure logger
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    environment = TaxiObservationWrapper(gym.make('TaxiGoal-v0'))

    environment.reset()
    environment.render()

    # Configure and create the agent
    cfg = Configuration(classifier_length=1,
                        number_of_possible_actions=6,
                        epsilon=0.999,
                        do_ga=False,
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=taxi_metrics,
                        do_action_planning=True,
                        action_planning_frequency=50)

    logging.info(cfg)

    # Explore the environment
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(environment, 1000)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(environment, 10)

    for metric in exploit_metric:
        logging.info(metric)
