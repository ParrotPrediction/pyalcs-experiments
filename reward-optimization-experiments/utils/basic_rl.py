import random

import numpy as np


def qlearning(env, episodes, init_Q, epsilon, learning_rate, discount_factor,
              perception_to_state_mapper=lambda p: int(p)):

    Q = np.copy(init_Q)

    for i in range(1, episodes):
        state = perception_to_state_mapper(env.reset())
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, info = env.step(action)
            next_state = perception_to_state_mapper(next_state)

            if next_state is not None:
                discounted = np.max(Q[next_state, :])
            else:
                discounted = 0

            Q[state, action] = Q[state, action] + learning_rate * (
                reward + discount_factor * discounted - Q[state, action])

            state = next_state

    return Q


def rlearning(env, episodes, init_R, epsilon, learning_rate, zeta,
              init_rho=0,
              perception_to_state_mapper=lambda p: int(p)):

    R = np.copy(init_R)
    rho = init_rho

    for i in range(1, episodes):
        state = perception_to_state_mapper(env.reset())
        was_greedy = False
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(R[state, :])
                was_greedy = True

            next_state, reward, done, info = env.step(action)
            next_state = perception_to_state_mapper(next_state)

            if next_state is not None:
                discounted = np.max(R[next_state, :])
            else:
                discounted = 0

            R[state, action] = R[state, action] + learning_rate * (
                reward - rho + discounted - R[state, action])

            if was_greedy:
                rho = rho + zeta * (reward + np.max(R[next_state, :]) - discounted - rho)

            state = next_state

    return R, rho
