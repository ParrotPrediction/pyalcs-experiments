import random
from timeit import default_timer as timer

import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
import lcs.agents.yacs as yacs
import numpy as np


def dynaq(env, episodes, Q, MODEL,
          epsilon, learning_rate, gamma, planning_steps,
          knowledge_fcn,
          metrics_trial_freq,
          perception_to_state_mapper=lambda p: int(p), ):
    # metrics
    metrics = np.zeros((4, episodes))  # steps, model_size, time, knowledge

    for i in range(episodes):
        episode_steps = 0
        past_state = perception_to_state_mapper(env.reset())
        done = False

        start_ts = timer()

        while not done:
            # q-learning
            if random.uniform(0, 1) < epsilon:
                past_action = env.action_space.sample()
            else:
                past_action = np.argmax(Q[past_state, :])

            state, reward, done, info = env.step(past_action)
            state = perception_to_state_mapper(state)

            if state is not None:
                discounted = np.max(Q[state, :])
            else:
                discounted = 0

            Q[past_state, past_action] += learning_rate * (
                reward + gamma * discounted - Q[past_state, past_action])

            # model update
            if past_state not in MODEL:
                MODEL[past_state] = {}

            if past_action not in MODEL[past_state]:
                MODEL[past_state][past_action] = {}

            MODEL[past_state][past_action] = (state, reward)

            # planning
            for _ in range(planning_steps):
                s = random.choice(list(MODEL.keys()))
                a = random.choice(list(MODEL[s].keys()))

                (next_s, r) = MODEL[s][a]

                discounted = np.max(Q[next_s, :])
                Q[s, a] += learning_rate * (r + gamma * discounted - Q[s, a])

            # Next step
            past_state = state
            episode_steps += 1

        end_ts = timer()

        # collect metrics
        if i % metrics_trial_freq == 0:
            metrics[0, i] = episode_steps
            metrics[1, i] = sum(
                [len(actions) for state, actions in MODEL.items()])
            metrics[2, i] = end_ts - start_ts
            metrics[3, i] = knowledge_fcn(MODEL, env)

    return Q, MODEL, metrics


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
