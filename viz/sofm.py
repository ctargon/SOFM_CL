#
# sofm.py
# implementation of self organizing feature map
#

import numpy as np

class sofm:
	def __init__(self, dims, lr, iters, n_features):
		self.dims = dims
		self.init_lr = lr
		self.lr = self.init_lr
		self.iters = iters

		self.n_features = n_features

		self.init_radius = max(dims[0], dims[1]) / 2
		self.radius = self.init_radius
		# self.weights = np.random.random((dims[0], dims[1], self.n_features))
		self.weights = np.random.random((dims[0] * dims[1], self.n_features))

		self.time_delay = iters / np.log(self.init_radius)

	def find_bmu(self, x):
		min_dist = np.iinfo(np.int32).max
		bmu_i = np.array([0,0])
		# for i in range(self.dims[0]):
		# 	for j in range(self.dims[1]):
		# 		w = self.weights[i, j, :].reshape(1, self.n_features)
		# 		dist = np.sum(np.sqrt(np.power(w - x, 2)))

		# 		if dist < min_dist:
		# 			min_dist = dist
		# 			bmu_i = np.array([i,j])
		delta = self.weights - x.transpose()
		dists = np.sum((delta)**2, axis=1).reshape(self.dims[0], self.dims[1])

		bmu_i = np.argmin(dists)

		#bmu = self.weights[bmu_i[0], bmu_i[1], :].reshape(1, self.n_features)
		i = int(bmu_i) / self.dims[0]
		j = int(bmu_i) % self.dims[0]

		return np.array([i,j])#bmu, bmu_i

	def decay_neighborhood(self, i):
		return self.init_radius * np.exp(-float(i) / self.time_delay)

	def decay_lr(self, i):
		return self.init_lr * np.exp(-float(i) / self.iters)

	def get_influence(self, d):
		return np.exp(-d / (2 * (self.radius ** 2)))

	def train_step(self, x, i):
		bmu_i = self.find_bmu(x)
		self.radius = self.decay_neighborhood(i)
		self.lr = self.decay_lr(i)

		for i in range(self.dims[0]):
			for j in range(self.dims[1]):
				w = self.weights[(i * self.dims[0]) + j, :].reshape(1, self.n_features)
				dist_from_bmu = np.sum(np.sqrt(np.power(np.array([i,j]) - bmu_i, 2)))

				if dist_from_bmu <= self.radius:
					influence = self.get_influence(dist_from_bmu)
					new_w = w + (self.lr * influence * (x.reshape(1, self.n_features) - w))
					self.weights[(i * self.dims[0]) + j, :] = new_w.reshape(1, self.n_features) 

		
