def grid_transitions(grid_size):
    MAX_POS = grid_size
    LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3

    def _handle_state(state):
        moves = []
        (x, y) = state

        # handle inner rectangle - 4 actions available
        if 1 < x < MAX_POS and 1 < y < MAX_POS:
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        # handle bounds (except corners) - 3 actions available
        if x == 1 and y not in [1, MAX_POS]:  # left bound
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        if x == MAX_POS and y not in [1, MAX_POS]:  # right bound
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        if x not in [1, MAX_POS] and y == 1:  # lower bound
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))

        if x not in [1, MAX_POS] and y == MAX_POS:  # upper bound
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        # handle corners - 2 actions available
        if x == 1 and y == 1:  # left-down
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))

        if x == 1 and y == MAX_POS:  # left-up
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        if x == MAX_POS and y == 1:  # right-down
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))

        return moves

    transitions = []
    for x in range(1, MAX_POS + 1):
        for y in range(1, MAX_POS + 1):
            transitions += _handle_state((x, y))

    return transitions


def print_cl(cl):
    action = None
    if cl.action == 0:
        action = '⬅'
    if cl.action == 1:
        action = '➡'
    if cl.action == 2:
        action = '⬆'
    if cl.action == 3:
        action = '⬇'
    print(f"{cl.condition} - {action} - {cl.effect} "
          f"[fit: {cl.fitness:.3f}, r: {cl.r:.2f}]")
