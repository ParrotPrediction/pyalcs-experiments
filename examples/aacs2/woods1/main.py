import gym

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lcs.agents import EnvironmentAdapter
from lcs.metrics import population_metrics

from lcs.agents.aacs2 import AACS2, Configuration

# Logger
import logging

logging.basicConfig(level=logging.INFO)

import gym_woods

woods1 = gym.make('Woods1-v0')
woods2 = gym.make('Woods2-v0')
woods14 = gym.make('Woods14-v0')


def common_metrics(agent, env):
    pop = agent.get_population()

    metrics = {
        'agent': agent.__class__.__name__,
        'reliable': len([cl for cl in pop if cl.is_reliable()]),
    }

    if hasattr(agent, 'rho'):
        metrics['rho'] = agent.rho
    else:
        metrics['rho'] = 0

    metrics.update(population_metrics(pop, env))
    return metrics


if __name__ == '__main__':

    # Experiments with exploration, then exploitation
    cfg = Configuration(8,
                        8,
                        do_ga=True,
                        zeta=0.001,
                        epsilon=0.99,
                        user_metrics_collector_fcn=common_metrics,
                        biased_exploration_prob=0.5,
                        metrics_trial_frequency=1)

    agent = AACS2(cfg)
    population_explr, metrics_explr = agent.explore(woods1, 50)

    # Exploit
    agent_explt = AACS2(cfg=cfg, population=population_explr)
    population_explt, metrics_explt = agent_explt.exploit(
        woods1, 10)
