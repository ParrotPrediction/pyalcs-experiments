import logging

import click
import gym
# noinspection PyUnresolvedReferences
import gym_mountain_car
import mlflow
import numpy as np
from lcs.agents.acs2 import Configuration, ACS2
from lcs.metrics import population_metrics

from observation_wrapper import MountainCarObservationWrapper

logging.basicConfig(level=logging.INFO)


def mc_metrics(agent, env):
    pop = agent.population
    metrics = {
        'avg_fitness': np.mean([cl.fitness for cl in pop if cl.is_reliable()])}
    metrics.update(population_metrics(pop, env))

    return metrics


@click.command(help="Perform experiments over MountainCar environment")
@click.option("--environment", type=click.STRING, default="MountainCar-v0")
@click.option("--trials", type=click.INT, default=1000,
              help="Exploration trials")
@click.option("--position-bins", type=click.INT, default=10)
@click.option("--velocity-bins", type=click.INT, default=20)
@click.option("--biased-exploration-prob", type=click.FLOAT, default=0.00,
              help='Probability of executing biased exploration')
@click.option("--decay", type=click.BOOL, default=False,
              help="Decay in exploration phase")
@click.option("--gamma", type=click.FLOAT, default=0.95)
def run(environment, trials, position_bins, velocity_bins,
        biased_exploration_prob, decay, gamma):

    bins = [position_bins, velocity_bins]

    with mlflow.start_run():
        logging.info("Initializing environment...")
        env = MountainCarObservationWrapper(gym.make(environment), bins)
        env._max_episode_steps = 1000

        cfg = Configuration(
            classifier_length=2,
            number_of_possible_actions=3,
            epsilon=0.9999,
            biased_exploration=biased_exploration_prob,
            beta=0.2,
            gamma=gamma,
            theta_as=50,
            theta_exp=100,
            theta_ga=50,
            do_ga=True,
            mu=0.03,
            chi=0.0,
            metrics_trial_frequency=5,
            user_metrics_collector_fcn=mc_metrics,
            use_mlflow=True)

        logging.info(f"Running {trials} experiments with"
                     f"optional decay (decay = {decay})")
        agent = ACS2(cfg)
        agent.explore(env, trials, decay=decay)


if __name__ == "__main__":
    run()
