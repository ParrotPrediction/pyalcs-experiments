from lcs.agents.xcs import XCS, Configuration
import pandas as pd


def xcs_metrics(xcs: XCS, environment):
    return {
        'population': len(xcs.population),
        'numerosity': sum(cl.numerosity for cl in xcs.population)
    }


def parse_results(metrics, cfg):
    df = pd.DataFrame(metrics)
    df['trial'] = df.index * cfg.metrics_trial_frequency
    df.set_index('trial', inplace=True)
    return df


def parse_results_exploit(metrics, cfg, explore_trials):
    df = pd.DataFrame(metrics)
    df['trial'] = df.index * cfg.metrics_trial_frequency + explore_trials
    df.set_index('trial', inplace=True)
    return df


def avg_experiment(maze, cfg, number_of_tests=1, explore_trials=4000, exploit_metrics=1000):
    test_metrics =[]
    for i in range(number_of_tests):
        print(f'Executing {i} experiment')
        test_metrics.append(start_single_experiment(maze, cfg, explore_trials, exploit_metrics))
    return pd.concat(test_metrics).groupby(['trial']).mean()


def start_single_experiment(maze, cfg, explore_trials=4000, exploit_metrics=1000):
    agent = XCS(cfg)
    explore_population, explore_metrics = agent.explore(maze, explore_trials, True)
    agent = XCS(cfg=cfg, population=explore_population)
    exploit_population, exploit_metrics = agent.exploit(maze, exploit_metrics)

    df = parse_results(explore_metrics, cfg)
    df_exploit = parse_results_exploit(exploit_metrics, cfg, explore_trials)
    df = df.append(df_exploit)
    return df
