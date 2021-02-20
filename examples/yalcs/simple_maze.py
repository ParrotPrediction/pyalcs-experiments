import gym
# noinspection PyUnresolvedReferences
import gym_yacs_simple_maze
import logging
from lcs.agents.yacs.yacs import Configuration, YACS, Effect

logging.basicConfig(level=logging.INFO)


def _calculate_knowledge(pop, env):
    all_transitions = 0
    covered_transitions = 0

    for s0, action_states in env.env.TRANSITIONS.items():
        for action_state in action_states:
            all_transitions += 1

            action = action_state.action.value
            s1 = action_state.state

            p0 = env.env._perception(s0)
            p1 = env.env._perception(s1)

            match_set = pop.form_match_set(p0)
            action_set = match_set.form_action_set(action)
            desired_effect = Effect.diff(p0, p1)

            if any(cl.effect == desired_effect for cl in action_set):
                covered_transitions += 1

    return covered_transitions / all_transitions


def _present_classifiers(pop, env):
    for s0, action_states in env.env.TRANSITIONS.items():
        p0 = env.env._perception(s0)
        print(f"\nState: [{s0}] ({''.join(list(map(str, p0)))})")
        match_set = pop.form_match_set(p0)
        for cl in match_set:
            print(f"\t{cl} ===> {cl.anticipation(p0)}")


def simple_maze_metrics(agent, env):
    pop_len = len(agent.population)
    return {
        'pop': pop_len,
        'oscillating': len([cl for cl in agent.population if cl.oscillating]),
        'knowledge': _calculate_knowledge(agent.population, env),
        'avg_r': sum(cl.r for cl in agent.population) / pop_len
    }


if __name__ == '__main__':
    env = gym.make('SimpleMaze-v0')

    cfg = Configuration(4, 4,
                        learning_rate=0.1,
                        discount_factor=0.8,
                        trace_length=3,
                        estimate_expected_improvements=True,
                        feature_possible_values=[2, 2, 2, 2],
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=simple_maze_metrics)

    agent = YACS(cfg)

    print("\n*** EXPLORE ***")
    pop, metrics = agent.explore(env, 500)

    print("\n*** DESIRABILITY VALUES ***")
    for p, de in agent.desirability_values.items():
        state = env.env._state_id(list(map(int, p)))
        print(f"{state}:\t{de}")

    _present_classifiers(pop, env)

    print("\n*** EXPLOIT ***")
    cfg.metrics_trial_frequency = 1
    exploiter = YACS(cfg, agent.population, agent.desirability_values)
    exploiter.exploit(env, 1)
