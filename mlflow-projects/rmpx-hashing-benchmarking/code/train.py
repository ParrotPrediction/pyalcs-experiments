import hashlib

import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer
import mlflow
from lcs.agents import EnvironmentAdapter

from agents import run_acs, run_acs2, run_yacs

mlflow.set_tracking_uri("http://localhost/mlflow")


class HashingRmpxAdapter(EnvironmentAdapter):
    def __init__(self, hash_name, modulo):
        self.hash_name = hash_name
        self.modulo = modulo

    def to_genotype(self, obs):
        hashed = []

        # hash all attributes except last one
        for i in [str(i).encode('utf-8') for i in obs[:-1]]:
            h = hashlib.new(self.hash_name)
            h.update(str(i).encode('utf-8'))
            hash = int(h.hexdigest(), 16)

            hashed.append(str(hash % self.modulo))

        if obs[-1] == 0.0:
            hashed.append('F')  # false anticipation
        else:
            hashed.append('T')  # true anticipation

        return hashed


def metrics_collector(agent, env):
    return {
        'pop': len(agent.population),
    }


def run(rmpx_size, trials, agent, hash_name, modulo):
    env = gym.make(f'real-multiplexer-{rmpx_size}bit-v0')
    env_adapter = HashingRmpxAdapter(hash_name=hash_name, modulo=modulo)

    model_checkpoint_freq = trials / 25
    common_cfg = {
        "classifier_length": rmpx_size + 1,
        "number_of_possible_actions": 2,
        "beta": 0.1,
        "environment_adapter": env_adapter,
        "user_metrics_collector_fcn": metrics_collector,
        "metrics_trial_frequency": 25,
        "use_mlflow": False
    }

    if agent == 'acs':
        return run_acs(env, trials, {**common_cfg, **{}})
    elif agent == 'acs2':
        return run_acs2(env, trials, {**common_cfg, **{
            'do_ga': False
        }})
    elif agent == 'acs2_ga':
        return run_acs2(env, trials, {**common_cfg, **{
            'do_ga': True
        }})
    elif agent == 'yacs':
        return run_yacs(env, trials, {**common_cfg, **{
            'trace_length': 3,
            'estimate_expected_improvements': False,
            'feature_possible_values': [{str(i) for i in
                                         range(modulo)}] * rmpx_size + [
                                           {'F', 'T'}],
        }})

    else:
        raise ValueError(f'Unknown agent: {agent}')


if __name__ == '__main__':
    pop, metrics = run(
        rmpx_size=3,
        trials=100,
        agent='acs2',
        hash_name='sha256',
        modulo=16)

    print(f"Len pop : {len(pop)}")
    for cl in pop:
        print(cl)
