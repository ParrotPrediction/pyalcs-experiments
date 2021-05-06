import pandas as pd
from lcs.agents.xncs import XNCS, Configuration


def cl_accuracy(cl, cfg):
    if cl.error < cfg.epsilon_0:
        return 1
    else:
        return cfg.alpha * pow(1 / (cl.error * cfg.epsilon_0), cfg.v)


def fraction_accuracy(xncs):
    action_sets_percentages = []
    for action in range(xncs.cfg.number_of_actions):
        action_set = xncs.population.generate_action_set(action)
        total_accuracy = 0
        most_numerous = action_set[0]
        for cl in action_set:
            total_accuracy += cl_accuracy(cl, xncs.cfg)
            if cl.numerosity > most_numerous.numerosity:
                most_numerous = cl
        action_sets_percentages.append(cl_accuracy(most_numerous, xncs.cfg) / total_accuracy)
    return sum(action_sets_percentages) / xncs.cfg.number_of_actions


def specificity(xncs, population):
    total_specificity = 0
    for cl in population:
        total_specificity += pow(2, cl.wildcard_number)
    return total_specificity / xncs.population.numerosity


def xcs_maze_metrics(xncs: XNCS, environment):
    return {
        'numerosity': xncs.population.numerosity,
        'population': len(xncs.population),
        'average_specificity': specificity(xncs, xncs.population),
        'fraction_accuracy': fraction_accuracy(xncs)
    }


def avg_experiment(maze, cfg, number_of_tests=1, trials=3000):
    test_metrics =[]
    for i in range(number_of_tests):
        print(f'Executing {i} experiment')
        test_metrics.append(start_single_experiment(maze, cfg, trials))
    return pd.concat(test_metrics).groupby(['trial']).mean()


def start_single_experiment(maze, cfg, trials=3000):
    agent = XNCS(cfg)
    _, current_metrics = agent.explore(maze, trials, True)
    df = parse_results(current_metrics, cfg)
    return df


def parse_results(metrics, cfg):
    df = pd.DataFrame(metrics)
    df['trial'] = df.index * cfg.metrics_trial_frequency
    df.set_index('trial', inplace=True)
    return df
