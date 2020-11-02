from collections import namedtuple

StateAction = namedtuple('StateAction', 'id state action')


def get_all_state_action(state_to_actions):
    state_action = []

    idx = 1
    for state, actions in state_to_actions.items():
        if len(actions) > 0:
            for action in actions:
                state_action.append(StateAction(idx, state, action))
                idx += 1

    return state_action
