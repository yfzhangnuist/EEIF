from .opt_forest import OptForest
from .sampling import VSSampling
from .opt import E2LSH
from .opt import AngleLSH
from scipy.spatial import distance

class EEIF(OptForest):
	def __init__(self, lsh_family='L2OPT', num_trees=100, threshold=403, branch=0, granularity=1):
		if lsh_family == 'ALOPT':
			OptForest.__init__(self, num_trees, VSSampling(num_trees), AngleLSH(), threshold, branch, distance.cosine, granularity)
		elif lsh_family == 'L1OPT':
			OptForest.__init__(self, num_trees, VSSampling(num_trees), E2LSH(norm=1), threshold, branch, distance.cityblock, granularity)
		else:
			OptForest.__init__(self, num_trees, VSSampling(num_trees), E2LSH(norm=2), threshold, branch, distance.euclidean, granularity)