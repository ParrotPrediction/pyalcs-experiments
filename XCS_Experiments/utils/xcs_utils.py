from lcs.agents.xcs import XCS, Configuration
import pandas as pd
from xcs.scenarios import Scenario
from xcs.bitstrings import BitString
import numpy as np
# environment setup
import gym
# noinspection PyUnresolvedReferences
import gym_maze
import random
import numpy as np
from lcs.agents.xcs import ClassifiersList
from lcs.agents.xcs import Classifier
from lcs.agents.xcs import Condition


def maze_knowledge(population, environment):
    transitions = environment.env.get_all_possible_transitions()

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for start, action, end in transitions:
        p0 = environment.env.maze.perception(*start)
        if any([True for cl in population
                if predicts_successfully(cl, p0, action)]):
            nr_correct += 1
    return nr_correct / len(transitions) * 100.0


def predicts_successfully(cl: Classifier, p0, action):
    if cl.does_match(p0):
        if cl.action == action:
            return True
    return False


def cl_accuracy(cl, cfg):
    if cl.error < cfg.epsilon_0:
        return 1
    else:
        return cfg.alpha * pow(1 / (cl.error * cfg.epsilon_0), cfg.v)


def specificity(xncs, population):
    total_specificity = 0
    for cl in population:
        total_specificity += pow(2, cl.wildcard_number) * cl.numerosity
    return total_specificity / xncs.population.numerosity


def xcs_metrics(xcs: XCS, environment):
    return {
        'population': len(xcs.population),
        'numerosity': sum(cl.numerosity for cl in xcs.population),
        'average_specificity': specificity(xcs, xcs.population),
    }


def xcs_maze_metrics(xcs: XCS, environment):
    return {
        'population': len(xcs.population),
        'numerosity': sum(cl.numerosity for cl in xcs.population),
        'average_specificity': specificity(xcs, xcs.population),
        'knowledge': maze_knowledge(xcs.population, environment),
    }


def XCS_classifier(situation, cfg):
    generalized = []
    for i in range(len(situation)):
        if np.random.rand() > cfg.covering_wildcard_chance:
            generalized.append(cfg.classifier_wildcard)
        else:
            generalized.append(situation[i])

    return Classifier(condition=Condition(generalized),
                      action=random.randrange(0, cfg.number_of_actions),
                      time_stamp=0,
                      cfg=cfg)

def XCS_population(maze, cfg):
    classifiers_list = ClassifiersList(cfg)
    while classifiers_list.numerosity < cfg.max_population:
        situation = maze.reset()
        classifiers_list.insert_in_population(XCS_classifier(situation, cfg))
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


def avg_experiment(maze, cfg, number_of_tests=1, explore_trials=4000, exploit_trials=1000, pre_generate=False):
    test_metrics = []
    for i in range(number_of_tests):
        print(f'Executing {i} experiment')
        if pre_generate:
            population = XCS_population(maze, cfg)
        else:
            population = None
        test_metrics.append(start_single_experiment(maze, cfg, explore_trials, exploit_trials, population))
    return pd.concat(test_metrics).groupby(['trial']).mean()


def start_single_experiment(maze, cfg, explore_trials=4000, exploit_metrics=1000, population=None):
    agent = XCS(cfg, population)
    explore_population, explore_metrics = agent.explore(maze, explore_trials, False)
    exploit_population, exploit_metrics = agent.exploit(maze, exploit_metrics)

    df = parse_results(explore_metrics, cfg)
    df_exploit = parse_results_exploit(exploit_metrics, cfg, explore_trials)
    df = df.append(df_exploit)
    return df


def other_avg_experiment(maze, algorithm, number_of_tests=1, explore_trials=4000, exploit_trials=1000):
    test_metrics = []
    for i in range(number_of_tests):
        print(f'Executing {i} experiment')
        test_metrics.append(
            other_start_single_test_explore(maze, algorithm, explore_trials, exploit_trials)
        )
    return pd.concat(test_metrics).groupby(['trial']).mean()


def other_start_single_test_explore(maze, algorithm, explore_trials, exploit_trials):
    maze.reset()
    tmp = algorithm.exploration_probability
    model = algorithm.new_model(maze)

    steps = []
    pop = []
    numerosity = []
    for i in range(explore_trials):
        maze.reset()
        model.run(maze, learn=True)
        if i % 100 == 0:
            steps.append(maze.steps)
            pop.append(len(model))
            numerosity.append(sum(rule.numerosity for rule in model))
    algorithm.exploration_probability = 0
    for i in range(explore_trials, exploit_trials + explore_trials):
        maze.reset()
        model.run(maze, learn=True)
        if (i + explore_trials) % 100 == 0:
            steps.append(maze.steps)
            pop.append(len(model))
            numerosity.append(sum(rule.numerosity for rule in model))
    algorithm.exploration_probability = tmp
    df = pd.DataFrame(data={'steps_in_trial': steps,
                            'population': pop,
                            'numerosity': numerosity})
    df['trial'] = df.index * 100
    df.set_index('trial', inplace=True)
    return df


class MazeScenario(Scenario):

    def __init__(self, input_size=8):
        np.random.seed(1)
        self.input_size = input_size
        self.maze = gym.make('Maze4-v0')
        self.possible_actions = (0, 1, 2, 3, 4, 5, 6, 7)
        self.done = False
        self.state = None
        self.reward = 0
        self.state = self.maze.reset()
        self.steps_array = []
        self.steps = 0

    def reset(self):
        self.done = False
        self.steps = 0
        self.state = self.maze.reset()
        return self.state

    # XCS Hosford42 functions
    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def more(self):
        if self.done:
            return False
        return True

    def sense(self):
        no_reward_state = []
        for char in self.state:
            if char == '1' or char == '0':
                no_reward_state.append(char)
            else:
                no_reward_state.append('1')
        return BitString(''.join(no_reward_state))

    def execute(self, action):
        self.steps += 1
        raw_state, step_reward, done, _ = self.maze.step(action)
        self.state = raw_state
        self.reward = step_reward
        self.done = done
        return self.reward

    # XCS Pyalcs functions
    def step(self, action):
        raw_state, step_reward, done, _ = self.maze.step(action)
        self.state = raw_state
        self.reward = step_reward
        self.done = done
        return raw_state, self.reward, self.done, _

