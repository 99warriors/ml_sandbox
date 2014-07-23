import distributions

class ArrayWithCounts(object):

    def __init__(self, vals):
        """
        vals is list
        """
        self.counts = {}
        self.vals = [np.nan for i in xrange(vals)]
        for (i, val) in enumerate(vals):
            self[i] = val

    def pop(self, idx):
        if not np.isnan(self.vals[idx]):
            self.vals[idx] = np.nan
            current_count = self.counts[self.vals[idx]]
            assert current_count > 0
            self.counts[self.vals[idx]] = current_count - 1
            if self.counts[self.vals[idx]] == 0:
                self.counts.pop(self.vals[idx], None)

    def get_all_items(self):
        return self.counts.keys()

    def get_count(self, val):
        return self.counts.get(val, 0)

    def get_total_count(self):
        return np.sum([count for (key, count) in self.counts.iteritems()])

    def __getitem__(self, idx):
        return self.vals[idx]

    def __setitem__(self, idx, val):
        self.pop(idx)
        self.vals[idx] = val
        try:
            self.counts[val] += 1
        except KeyError:
            self.counts[val] = 1
        
        

def get_dirichlet_process_mixture_direct_representation_gibbs_samples(num_steps, base_distribution, alpha, likelihood_function, data):
    """
    gibbs sampling for the direct representation of DP mixture model, where each data is associated with a draw from DP
    """
    N = len(data)
    thetas = ArrayWithCounts([base_distribution.sample() for i in xrange(N)])
    for step in xrange(num_steps):
        for n in xrange(N):
            thetas.pop(n)
            pis, categorical_distributions = itertools.izip([(float(thetas.get_count(theta)) * likelihood_function(data[n], theta) / (N + alpha - 1), distributions.CategoricalDistribution(val)) for theta in thetas.get_all_items()])
            pis.append(alpha / (N + alpha - 1))
            
        
