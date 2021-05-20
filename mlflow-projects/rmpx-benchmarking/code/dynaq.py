import random
from timeit import default_timer as timer
import numpy as np
import mlflow
import tempfile
import logging
import dill


def _metrics_dict(metrics, i, reward):
    return {
        'knowledge': metrics[3, i],
        'pop': metrics[1, i],
        'trial': i,
        'generalization': 0,
        'perf_time': metrics[2, i],
        'steps_in_trial': metrics[0, i],
        'reward': reward
    }

def dynaq(env, episodes, Q, MODEL,
          epsilon, learning_rate, gamma, planning_steps,
          knowledge_fcn,
          metrics_trial_freq,
          model_checkpoint_freq: int = None,
          perception_to_state_mapper=lambda p: int(p),
          using_mlflow=False):
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

            if using_mlflow:
                mlflow.log_metrics(_metrics_dict(metrics, i, reward), i)

        # checkpoint model and metrics
        if model_checkpoint_freq:
            if i % model_checkpoint_freq == 0:
                prefix = f"-trial-{i}"
                with tempfile.TemporaryDirectory(prefix) as td:
                    logging.debug(f"checkpointing model to {td}")
                    model_path = f"{td}/model_q.dill"
                    metrics_path = f"{td}/metrics.dill"

                    dill.dump((MODEL, Q),
                              open(model_path, mode='wb'))

                    dill.dump(metrics, open(metrics_path, mode='wb'))

                    if using_mlflow:
                        mlflow.log_artifacts(td, f"{i}/")

    return Q, MODEL, metrics

