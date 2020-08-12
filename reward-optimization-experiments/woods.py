import gym
import gym_woods
import lcs.agents.acs2 as acs2
import pandas as pd
from utils import WoodsAdapter, metrics

import matplotlib.pyplot as plt

# %% Initialize the environment
woods = gym.make('Woods2-v0')
s = woods.reset()


# %% Executing experiments
def parse_metrics(metrics):
    lst = [[d['trial'], d['steps_in_trial'], d['rho'], d['population'], d['reliable']] for d in metrics]

    df = pd.DataFrame(lst, columns=['trial', 'steps_in_trial', 'rho', 'population', 'reliable'])
    # df = df.set_index('trial')
    # df['phase'] = df.index.map(
    #     lambda t: "explore" if t % 2 == 0 else "exploit")

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



# %% ACS2
acs2_cfg = acs2.Configuration(8, 8,
                              do_ga=True,
                              # environment_adapter=WoodsAdapter,
                              epsilon=0.9,
                              user_metrics_collector_fcn=metrics,
                              metrics_trial_frequency=1)
acs2_agent = acs2.ACS2(cfg=acs2_cfg)
# acs2_perf_df = avg_experiments(1, woods, acs2_agent, 1000)


population, metrics = acs2_agent.explore(woods, 20_000)
explore_metrics = parse_metrics(metrics)

reliable = [cl for cl in population if cl.is_reliable()]


exploiter = acs2.ACS2(acs2_cfg, population)
pop2, met2 = acs2_agent.exploit(woods, 50)

exploit_metrics = parse_metrics(met2)

# %% Plots
fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2)

# plot 1
explore_metrics['population'].plot(ax=ax1, label='population')
explore_metrics['reliable'].plot(ax=ax1, label='reliable')

ax1.set_title('Exploration classifiers')
ax1.legend()

# plot 2
exploit_metrics['steps_in_trial'].plot(ax=ax2)
ax2.set_title('Exploitation steps')


plt.show()

# %% exploit plots
exploit_metrics['steps_in_trial'].plot()
plt.show()
