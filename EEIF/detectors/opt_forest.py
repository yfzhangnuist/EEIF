import numpy as np
import copy as cp
import random
from tqdm import trange
from . import opt_tree as Ot
from joblib import Parallel, delayed
import copy as cp

class OptForest:
	def __init__(self, num_trees, sampler, lsh_family, threshold, branch, distance, granularity=1):
		self._num_trees = num_trees
		self._sampler = sampler
		self._lsh_family = lsh_family
		self._granularity = granularity
		self._trees = []
		self.threshold = threshold
		self.branch = branch
		self.distance = distance

	def display(self):
		for t in self._trees:
			t.display()

	def fit(self, data):
		self.build(data)

	def build(self, data):
		indices = range(len(data))
		data = np.c_[indices, data]
		self._trees = []
		# Sampling data
		sampled_datas = []
		for _ in range(self._num_trees):
			if data.shape[0] >= 512:
				sampled_rows = np.random.choice(data.shape[0], 512, replace=False)
			else:
				sampled_rows = np.arange(data.shape[0])
			sample = np.array([data[row] for row in sampled_rows])
			sampled_datas.append(sample)
		sampled_datas = np.array(sampled_datas)

		# Build LSH instances based on the given data
		lsh_instances = []
		for i in range(self._num_trees):
			transformed_data = data
			if self._sampler._bagging != None:
				transformed_data = self._sampler._bagging_instances[i].get_transformed_data(data)	
			self._lsh_family.fit(transformed_data)
			lsh_instances.append(cp.deepcopy(self._lsh_family))

		# Build LSH trees
		for i in trange(self._num_trees):
			sampled_data = sampled_datas[i]
			tree = Ot.HierTree(lsh_instances[i], self.threshold, self.branch, self.distance)
			tree.build(sampled_data)
			self._trees.append(tree)

	def decision_function(self, data):
		indices = range(len(data))
		# Uncomment the following code for continuous data
		data = np.c_[indices, data]
		depths=[]
		data_size = len(data)
		
		def process_batch(batch, data, sampler, trees, granularity):
			batch_depths = []
			for i in batch:
				d_depths = []
				for j in range(len(trees)):
					transformed_data = data[i]
					if sampler._bagging != None:
						transformed_data = sampler._bagging_instances[j].get_transformed_data(np.mat(data[i])).A1
					d_depths.append(trees[j].predict(granularity, transformed_data))
				batch_depths.append(d_depths)
			return batch_depths

		batch_size = 3000 
		batches = [range(i, min(i + batch_size, data_size)) for i in range(0, data_size, batch_size)]

		# Process each batch in parallel
		depths = []
		if data_size>10000:
			nn_jobs=min(data_size//3000+1,10)
		else:
			nn_jobs=1
		for batch_depths in Parallel(n_jobs=nn_jobs)(delayed(process_batch)(batch, data, self._sampler, self._trees, self._granularity) for batch in batches):
			depths.extend(batch_depths)

		avg_depths=[]
		for i in range(data_size):
			depth_avg = 0.0
			for j in range(self._num_trees):
				depth_avg += depths[i][j]
			depth_avg /= self._num_trees
			avg_depths.append(depth_avg)

		avg_depths = np.array(avg_depths)
		return -1.0*avg_depths


	def get_avg_branch_factor(self):
		sum = 0.0
		for t in self._trees:
			sum += t.get_avg_branch_factor()
		return sum/self._num_trees		
