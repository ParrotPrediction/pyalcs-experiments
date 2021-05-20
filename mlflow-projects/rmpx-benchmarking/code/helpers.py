from lcs import Perception
from lcs.agents import EnvironmentAdapter

rmpx_utils = None


class RealMultiplexerAdapter(EnvironmentAdapter):
    @classmethod
    def to_genotype(cls, obs):
        return rmpx_utils.discretize(obs, _type=str)


def rmpx_knowledge(population, env):
    reliable = [c for c in population if c.is_reliable()]
    nr_correct = 0

    for start, action, end in rmpx_utils.get_transitions():
        p0 = Perception([str(el) for el in start])
        p1 = Perception([str(el) for el in end])

        if any([True for cl in reliable if
                cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return nr_correct / len(rmpx_utils.get_transitions())


def generalization_score(pop):
    wildcards = sum(1 for cl in pop for cond in cl.condition if
                    cond == '#' or (
                            hasattr(cond, 'symbol') and cond.symbol == '#'))
    all_symbols = sum(len(cl.condition) for cl in pop)
    return wildcards / all_symbols


def rmpx_metrics_collector(agent, env):
    population = agent.population
    return {
        'pop': len(population),
        'knowledge': rmpx_knowledge(population, env),
        'generalization': generalization_score(population)
    }

# DynaQ helpers
def rmpx_perception_to_int(p0, discretize=True):
    if discretize:
        p0 = rmpx_utils.discretize(p0)

    return rmpx_utils.state_mapping_inv[tuple(p0)]

def dynaq_rmpx_knowledge_calculator(model, env):
    all_transitions = 0
    nr_correct = 0

    for p0, a, p1 in rmpx_utils.get_transitions():
        s0 = rmpx_perception_to_int(p0, discretize=False)
        s1 = rmpx_perception_to_int(p1, discretize=False)

        all_transitions += 1
        if s0 in model and a in model[s0] and model[s0][a][0] == s1:
            nr_correct += 1

    return nr_correct / len(rmpx_utils.get_transitions())
