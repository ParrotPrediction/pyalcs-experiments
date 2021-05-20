import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
import lcs.agents.yacs as yacs
import numpy as np


def run_acs(return_data,
            run_id,
            env,
            classifier_length,
            possible_actions,
            learning_rate,
            environment_adapter,
            metrics_trial_freq,
            metrics_fcn, explore_trials):
    cfg = acs.Configuration(classifier_length, possible_actions,
                            beta=learning_rate,
                            environment_adapter=environment_adapter,
                            metrics_trial_frequency=metrics_trial_freq,
                            user_metrics_collector_fcn=metrics_fcn)

    agent = acs.ACS(cfg)
    pop, metrics = agent.explore(env, explore_trials)

    return_data[run_id] = (pop, metrics)


def run_acs2(return_data,
             run_id,
             env,
             classifier_length,
             possible_actions,
             learning_rate,
             environment_adapter,
             metrics_trial_freq,
             metrics_fcn,
             explore_trials,
             do_ga):
    cfg = acs2.Configuration(classifier_length, possible_actions,
                             beta=learning_rate,
                             environment_adapter=environment_adapter,
                             do_ga=do_ga,
                             metrics_trial_frequency=metrics_trial_freq,
                             user_metrics_collector_fcn=metrics_fcn)
    agent = acs2.ACS2(cfg)
    pop, metrics = agent.explore(env, explore_trials)

    return_data[run_id] = (pop, metrics)


def run_yacs(return_data,
             run_id,
             env,
             classifier_length,
             possible_actions,
             learning_rate,
             environment_adapter,
             metrics_trial_freq,
             metrics_fcn,
             explore_trials,
             trace_length,
             estimate_expected_improvements,
             feature_possible_values):
    cfg = yacs.Configuration(classifier_length, possible_actions,
                             learning_rate=learning_rate,
                             environment_adapter=environment_adapter,
                             trace_length=trace_length,
                             estimate_expected_improvements=estimate_expected_improvements,
                             feature_possible_values=feature_possible_values,
                             metrics_trial_frequency=metrics_trial_freq,
                             user_metrics_collector_fcn=metrics_fcn)
    agent = yacs.YACS(cfg)
    pop, metrics = agent.explore(env, explore_trials)

    return_data[run_id] = (pop, metrics)


def run_dynaq(return_data, run_id, env, **kwargs):
    q_init = np.zeros((kwargs['num_states'], kwargs['possible_actions']))
    model_init = {}  # maps state to actions to (reward, next_state) tuples

    Q, MODEL, metrics = dynaq(env,
                              episodes=kwargs['explore_trials'],
                              Q=q_init,
                              MODEL=model_init,
                              epsilon=0.5,
                              learning_rate=kwargs['learning_rate'],
                              gamma=0.9,
                              planning_steps=5,
                              knowledge_fcn=kwargs['knowledge_fcn'],
                              perception_to_state_mapper=kwargs[
                                  'perception_to_state_mapper'],
                              metrics_trial_freq=kwargs['metrics_trial_freq'])

    return_data[run_id] = (Q, MODEL, metrics)
