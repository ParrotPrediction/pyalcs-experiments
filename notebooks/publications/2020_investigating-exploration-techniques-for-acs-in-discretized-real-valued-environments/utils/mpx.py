import pandas as pd
from lcs.agents.acs2 import ACS2, Configuration


def parse_metrics(metrics):
    lst = [[d['trial'], d['reward'], d['population'], d['reliable']] for d in metrics]

    df = pd.DataFrame(lst, columns=['trial', 'reward', 'population', 'reliable'])
    # df = df.set_index('trial')
    df['phase'] = df.index.map(lambda t: "explore" if t % 2 == 0 else "exploit")

    return df



def start_single_experiment(env, trials, **kwargs):
    env.reset()
    cfg = Configuration(**kwargs)

    agent = ACS2(cfg)
    population, metrics = agent.explore_exploit(env, trials)

    metrics_df = parse_metrics(metrics)

    return population, metrics_df


def avg_experiments(n, env, trials, **kwargs):
    dfs = []
    print(f"{kwargs}\n")

    for i in range(n):
        print(f"Executing experiment {i}")
        _, df = start_single_experiment(env, trials, **kwargs)
        dfs.append(df)

    bar = pd.concat(dfs)
    perf_df = bar.groupby(['trial', 'phase']).mean().reset_index(level='phase')

    return perf_df
