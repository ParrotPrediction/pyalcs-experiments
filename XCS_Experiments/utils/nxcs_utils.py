import pandas as pd
import numpy as np
import random
from lcs.agents.xncs import XNCS, Configuration, Classifier, ClassifiersList, Effect
from lcs.agents.xcs import Condition


def maze_knowledge(population, environment):
    transitions = environment.env.get_all_possible_transitions()

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for start, action, end in transitions:
        p0 = environment.env.maze.perception(*start)
        p1 = environment.env.maze.perception(*end)
        if any([True for cl in population
                if predicts_successfully(cl, p0, action, p1)]):
            nr_correct += 1
    return nr_correct / len(transitions) * 100.0


def predicts_successfully(cl: Classifier, p0, action, p1):
    if cl.does_match(p0):
        if cl.action == action:
            if cl.effect is None:
                return False
            if cl.effect == Effect(p1):
                return True
    return False


def cl_accuracy(cl, cfg):
    if cl.error < cfg.epsilon_0:
        return 1
    else:
        return cfg.alpha * pow(1 / (cl.error * cfg.epsilon_0), cfg.v)


def fraction_accuracy(xncs: XNCS):
    return xncs.fraction_accuracy


def specificity(xncs, population):
    total_specificity = 0
    for cl in population:
        total_specificity += pow(2, cl.wildcard_number) * cl.numerosity
    return total_specificity / xncs.population.numerosity


def xncs_maze_metrics(xncs: XNCS, environment):
    return {
        'numerosity': xncs.population.numerosity,
        'population': len(xncs.population),
        'average_specificity': specificity(xncs, xncs.population),
        'fraction_accuracy': fraction_accuracy(xncs),
        'knowledge': maze_knowledge(xncs.population, environment),
    }

def xncs_woods_metrics(xncs: XNCS, environment):
    return {
        'numerosity': xncs.population.numerosity,
        'population': len(xncs.population),
        'average_specificity': specificity(xncs, xncs.population),
        'fraction_accuracy': fraction_accuracy(xncs),
    }


def xncs_metrics(xncs: XNCS, environment):
    return {
        'numerosity': xncs.population.numerosity,
        'population': len(xncs.population),
        'average_specificity': specificity(xncs, xncs.population),
        'fraction_accuracy': fraction_accuracy(xncs)
    }


def avg_experiment(maze, cfg, number_of_tests=1, explore_trials=3000, exploit_trials=1000, pre_generate=False):
    test_metrics =[]
    for i in range(number_of_tests):
        print(f'Executing {i} experiment')
        if pre_generate:
            population = XNCS_population(maze, cfg)
        else:
            population = None
        test_metrics.append(start_single_experiment(maze,
                                                    cfg,
                                                    explore_trials,
                                                    exploit_trials,
                                                    population))
    return pd.concat(test_metrics).groupby(['trial']).mean()


def start_single_experiment(maze, cfg, explore_trials, exploit_trials, population=None):
    agent = XNCS(cfg, population)
    _, explore_metrics = agent.explore(maze, explore_trials)
    _, exploit_metrics = agent.exploit(maze, exploit_trials)
    df = parse_results(explore_metrics, cfg)
    df_exploit = parse_results_exploit(exploit_metrics, cfg, explore_trials)
    df = df.append(df_exploit)
    return df


def XNCS_classifier(situation, cfg):
    generalized = []
    effect = []
    for i in range(len(situation)):
        if np.random.rand() > cfg.covering_wildcard_chance:
            generalized.append(cfg.classifier_wildcard)
        else:
            generalized.append(situation[i])
        effect.append(str(random.choice(situation)))
    cl = Classifier(cfg=cfg,
                    condition=Condition(generalized),
                    action=random.randrange(0, cfg.number_of_actions),
                    time_stamp=0,
                    effect=Effect(effect))
    return cl


def XNCS_population(maze, cfg):
    classifiers_list = ClassifiersList(cfg)
    while classifiers_list.numerosity < cfg.max_population:
        situation = maze.reset()
        classifiers_list.insert_in_population(XNCS_classifier(situation, cfg))
    return classifiers_list


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
