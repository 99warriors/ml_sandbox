import numpy as np
import itertools
import scipy.misc

class Distribution(object):

    def sample(self):
        raise NotImplementedError

    def loglik(self, x):
        raise NotImplementedError

class DeltaDistribution(distribution):

    def __init__(self, val):
        self.val = val

    def sample(self):
        return self.val

    def loglik(self, x):
        if x != self.val:
            return -np.inf
        else:
            return 0


class CategoricalDistribution(distribution):
    """
    discrete distribution over 0 ... len(pi)-1
    """
    def __init__(self, pi):
        self.pi = pi
        self.cum_pi = np.cumsum(self.pi)

    def sample(self):
        r = np.random.uniform(np.sum(self.pi))
        while True:
            if r < self.cum_pi[i]:
                return i
            i = i + 1
        assert False

    def loglik(self, x):
        return np.log(self.pi[x])


class MixtureDistribution(distribution):

    def __init__(self, pi, component_distributions):
        self.pi, self.component_distributions = pi, component_distributions
        self.mixing_categorical_distribution = CategoricalDistribution(pi)

    def sample(self):
        return self.component_distributions[self.mixing_categorical_distribution.sample()].sample()

    def loglik(self, x):
        scipy.misc.logsumexp([np.log(pi) + component_distribution.loglik(x) for (pi, component_distribution) in itertools.izip(self.pi, self.component_distributions)])
    
