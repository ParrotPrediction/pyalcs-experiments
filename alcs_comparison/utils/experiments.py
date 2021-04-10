import random
from timeit import default_timer as timer

import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
import lcs.agents.yacs as yacs
import numpy as np
import pandas as pd


# convert collected metrics into pandas dataframe
def parse_lcs_metrics(agent, metrics):
    data = [[agent, d['perf_time'], d['trial'], d['knowledge'], d['pop'],
             d['generalization']] for d in metrics]

    df = pd.DataFrame(
        data,
        columns=['agent', 'time', 'trial', 'knowledge', 'population',
                 'generalization'])

    return df


def parse_dyna_metrics(agent, metrics):
    (steps, model_size, time, knowledge) = metrics
    df = pd.DataFrame(metrics.T,
                      columns=['trial_steps', 'population', 'time',
                               'knowledge'])

    # add derived columns
    df['trial'] = df.index
    df['agent'] = agent
    df['generalization'] = 0

    return df


# Compute proportion of wildcards in classifier condition across all classifiers
def generalization_score(pop):
    wildcards = sum(1 for cl in pop for cond in cl.condition if
                    cond == '#' or (
                            hasattr(cond, 'symbol') and cond.symbol == '#'))
    all_symbols = sum(len(cl.condition) for cl in pop)
    return wildcards / all_symbols


def dynaq(env, episodes, num_states, num_actions,
          epsilon, learning_rate, gamma, planning_steps,
          knowledge_fcn,
          perception_to_state_mapper=lambda p: int(p)):

    # metrics
    metrics = np.zeros((4, episodes))  # steps, model_size, time, knowledge

    # init
    Q = np.zeros((num_states, num_actions))
    MODEL = {}  # maps state to actions to (reward, next_state) tuples

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
        metrics[0, i] = episode_steps
        metrics[1, i] = sum([len(actions) for state, actions in MODEL.items()])
        metrics[2, i] = end_ts - start_ts
        metrics[3, i] = knowledge_fcn(MODEL, env)

    return Q, MODEL, metrics


def run_acs(env, classifier_length, possible_actions, learning_rate, environment_adapter, metrics_trial_freq, metrics_fcn, explore_trials):
    cfg = acs.Configuration(classifier_length, possible_actions,
                            beta=learning_rate,
                            environment_adapter=environment_adapter,
                            metrics_trial_frequency=metrics_trial_freq,
                            user_metrics_collector_fcn=metrics_fcn)
    agent = acs.ACS(cfg)
    pop, metrics = agent.explore(env, explore_trials)
    
    return pop, metrics


def run_acs2(env, classifier_length, possible_actions, learning_rate, environment_adapter, metrics_trial_freq, metrics_fcn, explore_trials, do_ga):
    cfg = acs2.Configuration(classifier_length, possible_actions,
                             beta=learning_rate,
                             environment_adapter=environment_adapter,
                             do_ga=do_ga,
                             metrics_trial_frequency=metrics_trial_freq,
                             user_metrics_collector_fcn=metrics_fcn)
    agent = acs2.ACS2(cfg)
    pop, metrics = agent.explore(env, explore_trials)
    return pop, metrics


def run_yacs(env, classifier_length, possible_actions, learning_rate, environment_adapter, metrics_trial_freq, metrics_fcn, explore_trials, trace_length, feature_possible_values):
    cfg = yacs.Configuration(classifier_length, possible_actions,
                             learning_rate=learning_rate,
                             environment_adapter=environment_adapter,
                             trace_length=trace_length,
                             feature_possible_values=feature_possible_values,
                             metrics_trial_frequency=metrics_trial_freq,
                             user_metrics_collector_fcn=metrics_fcn)
    agent = yacs.YACS(cfg)
    pop, metrics = agent.explore(env, explore_trials)
    return pop, metrics


def run_dynaq(env, **kwargs):
    return dynaq(env,
                 episodes=kwargs['explore_trials'],
                 num_states=kwargs['num_states'],
                 num_actions=kwargs['possible_actions'],
                 epsilon=0.5,
                 learning_rate=kwargs['learning_rate'],
                 gamma=0.9,
                 planning_steps=5,
                 knowledge_fcn=kwargs['knowledge_fcn'],
                 perception_to_state_mapper=kwargs['perception_to_state_mapper'])


# Run single experiments using 5 algorithms. Return all population of classifiers and metrics
def run_experiment(common_params, acs_params={}, acs2_params={}, yacs_params={}, dynaq_params={}):
    pop_acs, metrics_acs = run_acs(**{**common_params, **acs_params})
    pop_acs2, metrics_acs2 = run_acs2(do_ga=False, **{**common_params, **acs2_params})
    pop_acs2ga, metrics_acs2ga = run_acs2(do_ga=True, **{**common_params, **acs2_params})
    pop_yacs, metrics_yacs = run_yacs(**{**common_params, **yacs_params})
    q_dynaq, model_dynaq, metrics_dynaq = run_dynaq(**{**common_params, **dynaq_params})
    
    metrics_df = pd.concat([
        parse_lcs_metrics('acs', metrics_acs),
        parse_lcs_metrics('acs2', metrics_acs2),
        parse_lcs_metrics('acs2_ga', metrics_acs2ga),
        parse_lcs_metrics('yacs', metrics_yacs),
        parse_dyna_metrics('dynaq', metrics_dynaq)
    ])
    metrics_df.set_index(['agent', 'trial'], inplace=True)
    
    return {
        'acs': pop_acs,
        'acs2': pop_acs2,
        'acs2_ga': pop_acs2ga,
        'yacs': pop_yacs,
        'dynaq': (q_dynaq, model_dynaq)
    }, metrics_df


# Execute multiple experiments and average metrics
def avg_experiments(func, n=1):
    dfs = []
    for i in range(n):
        print(f'Executing {i} experiment')
        _, metrics_df = func()
        dfs.append(metrics_df)
    
    return pd.concat(dfs).groupby(['agent', 'trial']).mean()
