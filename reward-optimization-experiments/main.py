# %% Imports
import gym
# noinspection PyUnresolvedReferences
import gym_fsw
import matplotlib.pyplot as plt
import pandas as pd
from utils import FSWAdapter, metrics
from lcs.agents.aacs2 import Configuration, AACS2
from lcs.agents.acs2ar import Configuration, ACS2AR

import matplotlib.pyplot as plt

# %% Initialize the environment
fsw = gym.make('fsw-10-v0')
s = fsw.reset()


# %% Executing experiments
def parse_metrics(metrics):
    lst = [[d['trial'], d['rho'], d['population'], d['reliable']] for d in
           metrics]

    df = pd.DataFrame(lst, columns=['trial', 'rho', 'population', 'reliable'])
    # df = df.set_index('trial')
    df['phase'] = df.index.map(
        lambda t: "explore" if t % 2 == 0 else "exploit")

    return df


def start_single_experiment(env, agent, trials):
    env.reset()

    population, metrics = agent.explore_exploit(env, trials)

    metrics_df = parse_metrics(metrics)

    return population, metrics_df


def avg_experiments(n, env, agent, trials):
    dfs = []

    for i in range(n):
        print(f"Executing experiment {i}")
        _, df = start_single_experiment(env, agent, trials)
        dfs.append(df)

    bar = pd.concat(dfs)
    perf_df = bar.groupby(['trial', 'phase']).mean().reset_index(level='phase')

    return perf_df


# %% AACS2
aacs2_cfg = Configuration(1, 2,
                          environment_adapter=Adapter,
                          user_metrics_collector_fcn=fsw_metrics,
                          metrics_trial_frequency=1)
aacs2_agent = AACS2(cfg=aacs2_cfg)
aacs2_perf_df = avg_experiments(1, fsw, aacs2_agent, 1300)

# %% Plot results for AACS2
fig, ax = plt.subplots()

aacs2_perf_df['rho'].plot(ax=ax)
ax.set_title('AACS2 rho')

plt.show()

# %% ACS2AR
acs2ar_cfg = Configuration(1, 2,
                           environment_adapter=Adapter,
                           user_metrics_collector_fcn=fsw_metrics,
                           metrics_trial_frequency=1)
acs2ar_agent = ACS2AR(cfg=acs2ar_cfg)

acs2ar_perf_df = avg_experiments(1, fsw, acs2ar_agent, 1300)

# %% Plot results for ACS2AR
fig, ax = plt.subplots()

acs2ar_perf_df['rho'].plot(ax=ax)
ax.set_title('ACS2AR rho')

plt.show()
