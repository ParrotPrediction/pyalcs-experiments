import mlflow

import helpers
from dynaq import dynaq
from helpers import RealMultiplexerAdapter, rmpx_metrics_collector, \
    rmpx_perception_to_int, dynaq_rmpx_knowledge_calculator
from rmpx_utils import RealMultiplexerUtils

import gym
import gym_multiplexer

import numpy as np

import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
import lcs.agents.yacs as yacs

import click

mlflow.set_tracking_uri("http://localhost/mlflow")


@click.command()
@click.option("--rmpx-size", type=click.Choice(['3', '6', '11', '20', '37']), required=True)
@click.option("--trials", type=click.INT, default=100)
@click.option("--bins", type=click.INT, default=4)
@click.option("--agent",
              type=click.Choice(['ACS', 'ACS2', 'ACS2GA', 'YACS', 'DynaQ'],
                                case_sensitive=False), required=True)
@click.option("--beta", type=click.FloatRange(0, 1), default=0.1)  # learning rate
def run(rmpx_size, trials, bins, agent, beta):
    rmpx_size = int(rmpx_size)

    mlflow.log_param("rmpx_size", rmpx_size)
    mlflow.log_param("trials", trials)
    mlflow.log_param("bins", bins)
    mlflow.log_param("agent", agent)

    # create environment
    env = gym.make(f'real-multiplexer-{rmpx_size}bit-v0')

    helpers.rmpx_utils = RealMultiplexerUtils(size=rmpx_size, bins=bins, env=env)

    # common params
    classifier_length = rmpx_size + 1
    possible_actions = 2
    learning_rate = beta
    metric_freq = trials / 100
    model_checkpoint_freq = trials / 25

    if agent == 'ACS':
        cfg = acs.Configuration(
            classifier_length=classifier_length,
            number_of_possible_actions=possible_actions,
            beta=learning_rate,
            environment_adapter=RealMultiplexerAdapter,
            metrics_trial_frequency=metric_freq,
            model_checkpoint_frequency=model_checkpoint_freq,
            user_metrics_collector_fcn=rmpx_metrics_collector,
            use_mlflow=True)
        agent = acs.ACS(cfg)
        agent.explore(env, trials)
    elif agent == 'ACS2':
        cfg = acs2.Configuration(
            classifier_length=classifier_length,
            number_of_possible_actions=possible_actions,
            beta=learning_rate,
            do_ga=False,
            environment_adapter=RealMultiplexerAdapter,
            metrics_trial_frequency=metric_freq,
            model_checkpoint_frequency=model_checkpoint_freq,
            user_metrics_collector_fcn=rmpx_metrics_collector,
            use_mlflow=True)
        agent = acs2.ACS2(cfg)
        agent.explore(env, trials)
    elif agent == 'ACS2GA':
        cfg = acs2.Configuration(
            classifier_length=classifier_length,
            number_of_possible_actions=possible_actions,
            beta=learning_rate,
            do_ga=True,
            environment_adapter=RealMultiplexerAdapter,
            metrics_trial_frequency=metric_freq,
            model_checkpoint_frequency=model_checkpoint_freq,
            user_metrics_collector_fcn=rmpx_metrics_collector,
            use_mlflow=True)
        agent = acs2.ACS2(cfg)
        agent.explore(env, trials)
    elif agent == 'YACS':
        cfg = yacs.Configuration(classifier_length, possible_actions,
                                 learning_rate=learning_rate,
                                 environment_adapter=RealMultiplexerAdapter,
                                 trace_length=3,
                                 estimate_expected_improvements=False,
                                 feature_possible_values=[bins] * rmpx_size + [
                                     2],
                                 metrics_trial_frequency=metric_freq,
                                 model_checkpoint_frequency=model_checkpoint_freq,
                                 user_metrics_collector_fcn=rmpx_metrics_collector,
                                 use_mlflow=True)
        agent = yacs.YACS(cfg)
        agent.explore(env, trials)
    elif agent == 'DynaQ':
        q_init = np.zeros(
            (len(helpers.rmpx_utils.state_mapping), possible_actions))
        model_init = {}  # maps state to actions to (reward, next_state) tuples

        dynaq(env,
              episodes=trials,
              Q=q_init,
              MODEL=model_init,
              epsilon=0.5,
              learning_rate=learning_rate,
              gamma=0.9,
              planning_steps=5,
              knowledge_fcn=dynaq_rmpx_knowledge_calculator,
              perception_to_state_mapper=rmpx_perception_to_int,
              metrics_trial_freq=metric_freq,
              model_checkpoint_freq=model_checkpoint_freq,
              using_mlflow=True)
    else:
        raise ValueError('Invalid agent')


if __name__ == '__main__':
    run()
