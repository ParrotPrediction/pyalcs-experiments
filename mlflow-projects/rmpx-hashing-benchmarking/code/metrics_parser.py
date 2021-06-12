import pandas as pd


def parse(metrics):
    data = [[d['perf_time'], d['trial'], d['reward'], d['pop']] for d in
            metrics]

    df = pd.DataFrame(
        data,
        columns=['time', 'trial', 'reward', 'population'])

    df = df.set_index('trial')
    df['mode'] = df.index.map(lambda t: "explore" if t % 2 == 0 else "exploit")

    return df
