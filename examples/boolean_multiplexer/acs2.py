import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer


from lcs.agents.acs2 import ACS2, Configuration

from examples.boolean_multiplexer import reliable_cl_exists, \
    MpxObservationWrapper


def mpx_metrics(agent, env):
    return {
        'population': len(agent.population),
        'reliable_cl_exists': reliable_cl_exists(env, agent.population, ctrl_bits=2)
    }


if __name__ == '__main__':
    # Load desired environment
    mp = MpxObservationWrapper(gym.make('boolean-multiplexer-6bit-v0'))

    # Create agent
    cfg = Configuration(classifier_length=mp.env.observation_space.n,
                        number_of_possible_actions=2,
                        do_ga=False,
                        metrics_trial_frequency=50,
                        user_metrics_collector_fcn=mpx_metrics)
    agent = ACS2(cfg)

    # Explore the environment
    population, explore_metrics = agent.explore(mp, 150)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metrics = agent.exploit(mp, 50)

    # See how it went
    for metric in explore_metrics:
        print(metric)
