from lcs.agents.xcs import XCS, Configuration
import pandas as pd
from xcs.scenarios import Scenario
from xcs.bitstrings import BitString
import numpy as np
# environment setup
import gym
# noinspection PyUnresolvedReferences
import gym_maze


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
    test_metrics = []
    for i in range(number_of_tests):
        print(f'Executing {i} experiment')
        test_metrics.append(start_single_experiment(maze, cfg, explore_trials, exploit_metrics))
    return pd.concat(test_metrics).groupby(['trial']).mean()


def start_single_experiment(maze, cfg, explore_trials=4000, exploit_metrics=1000):
    agent = XCS(cfg)
    explore_population, explore_metrics = agent.explore(maze, explore_trials, False)
    agent = XCS(cfg=cfg, population=explore_population)
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

