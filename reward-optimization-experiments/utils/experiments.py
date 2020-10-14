import logging

import lcs.agents.aacs2 as aacs2
import lcs.agents.acs2 as acs2
import pandas as pd
from metrics import parse_metrics


def avg_experiments(n, trials, env, params):
    dfs = []
    logging.debug(f"{params}\n")

    for i in range(n):
        print(f"Executing experiment {i}")
        _, _, _, m = run_experiments(env, trials, params)
        df = parse_metrics(m)
        dfs.append(df)

    all_dfs = pd.concat(dfs)
    agg_df = all_dfs.groupby(['agent', 'trial', 'phase']).mean().reset_index(
        level='phase')

    return agg_df


def run_experiments(env, trials, params):
    """
    Function running experiments in explore-exploit fashion using
    3 algorithms - ACS2, AACS2-v1, AACS2-v2
    """

    logging.info('Starting ACS2 experiments')
    acs2_cfg = acs2.Configuration(
        params['perception_bits'],
        params['possible_actions'],
        do_ga=params['do_ga'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        gamma=params['gamma'],
        environment_adapter=params['environment_adapter'],
        user_metrics_collector_fcn=params['user_metrics_collector_fcn'],
        biased_exploration_prob=params['biased_exploration_prob'],
        metrics_trial_frequency=params['metrics_trial_freq'])

    acs2_agent = acs2.ACS2(acs2_cfg)
    pop_acs2, metrics_acs2 = acs2_agent.explore_exploit(env, trials)

    logging.info('Starting AACS2-v1 experiments')
    aacs2v1_cfg = aacs2.Configuration(
        params['perception_bits'],
        params['possible_actions'],
        do_ga=params['do_ga'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        gamma=params['gamma'],
        zeta=params['zeta'],
        rho_update_version='1',
        environment_adapter=params['environment_adapter'],
        user_metrics_collector_fcn=params['user_metrics_collector_fcn'],
        biased_exploration_prob=params['biased_exploration_prob'],
        metrics_trial_frequency=params['metrics_trial_freq'])

    aacs2v1_agent = aacs2.AACS2(aacs2v1_cfg)
    pop_aacs2v1, metrics_aacs2v1 = aacs2v1_agent.explore_exploit(env, trials)

    logging.info('Starting AACS2-v2 experiments')
    aacs2v2_cfg = aacs2.Configuration(
        params['perception_bits'],
        params['possible_actions'],
        do_ga=params['do_ga'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        gamma=params['gamma'],
        zeta=params['zeta'],
        rho_update_version='2',
        environment_adapter=params['environment_adapter'],
        user_metrics_collector_fcn=params['user_metrics_collector_fcn'],
        biased_exploration_prob=params['biased_exploration_prob'],
        metrics_trial_frequency=params['metrics_trial_freq'])

    aacs2v2_agent = aacs2.AACS2(aacs2v2_cfg)
    pop_aacs2v2, metrics_aacs2v2 = aacs2v2_agent.explore_exploit(env, trials)

    # Join metrics together
    m = []
    m.extend(metrics_acs2)
    m.extend(metrics_aacs2v1)
    m.extend(metrics_aacs2v2)

    return pop_acs2, pop_aacs2v1, pop_aacs2v2, m
