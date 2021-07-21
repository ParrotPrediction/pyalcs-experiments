from typing import Dict

from lcs.metrics import population_metrics


def taxi_knowledge(population, environment) -> Dict:
    """
    Analyzes all possible transition in taxi environment and checks if there
    is a reliable classifier for it.

    Parameters
    ----------
    population
        list of classifiers
    environment
        taxi environment

    Returns
    -------
    Dict
        knowledge - percentage of transitions we are able to anticipate
            correctly (max 100)
    """
    transitions = environment.env.P

    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0
    nr_all = 0

    # For all possible destinations from each path cell
    for start in range(500):
        for action in range(6):
            local_transitions = transitions[start][action]

            prob, end, reward, done = local_transitions[0]

            if start != end:
                p0 = (str(start), )
                p1 = (str(end), )

                nr_all += 1

                if any([True for cl in reliable_classifiers
                        if cl.predicts_successfully(p0, action, p1)]):
                    nr_correct += 1

    return {
        'knowledge': nr_correct / nr_all * 100.0
    }


def taxi_metrics(agent, environment):
    population = agent.population
    return {
        'agent': population_metrics(population, environment),
        'knowledge': taxi_knowledge(population, environment)
    }
