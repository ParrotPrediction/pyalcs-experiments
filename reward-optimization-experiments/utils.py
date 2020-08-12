from lcs.agents import EnvironmentAdapter
from lcs.metrics import population_metrics


class FSWAdapter(EnvironmentAdapter):

    @classmethod
    def to_genotype(cls, phenotype):
        # Represent state as a single unicode character
        return chr(int(phenotype) + 65)

class WoodsAdapter(EnvironmentAdapter):

    @classmethod
    def to_genotype(cls, phenotype):
        result = []
        for el in phenotype:
            if el == 'F':
                result.extend(['1', '1', '0'])
            if el == 'G':
                result.extend(['1', '1', '1'])
            if el == 'O':
                result.extend(['0', '1', '0'])
            if el == 'Q':
                result.extend(['0', '1', '1'])
            if el == '.':
                result.extend(['0', '0', '0'])

        return result

def metrics(agent, env):
    pop = agent.get_population()

    metrics = {
        'reliable': len([cl for cl in pop if cl.is_reliable()]),
    }

    if hasattr(agent, 'rho'):
        metrics['rho'] = agent.rho
    else:
        metrics['rho'] = 0

    metrics.update(population_metrics(pop, env))
    return metrics
