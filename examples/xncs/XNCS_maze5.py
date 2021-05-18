from XCS_Experiments.utils.xcs_utils import *
from lcs.agents.xncs import Configuration, XNCS


env = MazeScenario(input_size=8)
env.maze.reset()
env.maze.render()

cfg = Configuration(number_of_actions=8,
                    max_population=400,
                    metrics_trial_frequency=100,
                    covering_wildcard_chance=0.9,
                    mutation_chance=1,
                    delta=0.1,
                    user_metrics_collector_fcn=xcs_metrics)

agent = XNCS(cfg)
population, metrics = agent.explore(env, 1000)
