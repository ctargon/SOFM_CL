#
# script to visualize weights of the sofm
#

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from PIL import Image
import re
import json
import sys
from GTEx import GTEx
import re

def find_bmu(weights, x, dims):
	min_dist = np.iinfo(np.int32).max
	bmu_i = np.array([0,0])
	# for i in range(self.dims[0]):
	# 	for j in range(self.dims[1]):
	# 		w = self.weights[i, j, :].reshape(1, self.n_features)
	# 		dist = np.sum(np.sqrt(np.power(w - x, 2)))

	# 		if dist < min_dist:
	# 			min_dist = dist
	# 			bmu_i = np.array([i,j])
	delta = weights - x.transpose()
	dists = np.sum((delta)**2, axis=1).reshape(dims[0], dims[1])

	bmu_i = np.argmin(dists)

	#bmu = self.weights[bmu_i[0], bmu_i[1], :].reshape(1, self.n_features)
	i = int(bmu_i) / dims[0]
	j = int(bmu_i) % dims[1]

	return np.array([i,j])#bmu, bmu_i


def load_data(num_samples_json, gtex_gct_flt):
	sample_count_dict = {}
	with open(num_samples_json) as f:
		sample_count_dict = json.load(f)

	idx = 0
	data = {}

	for k in sorted(sample_count_dict.keys()):
		data[k] = gtex_gct_flt[:,idx:(idx + int(sample_count_dict[k]))]
		idx = idx + int(sample_count_dict[k])

	return data


# read a csv or txt file that contains a name of a subset followed by a list of genes
def read_subset_file(file):
	with open(file, 'r') as f:
		content = f.readlines()

	# eliminate new line characters
	content = [x.strip() for x in content]

	# split on tabs or commas to create a sublist of set names and genes
	content = [re.split('\t|,| ', x) for x in content]

	# create a dictionary with keys subset names and values list of genes
	subsets = {}
	for c in content:
		subsets[c[0]] = c[1:]

	return subsets




if __name__ == '__main__':

	network_dimensions = [40,40]
	run = "opencl"
	n_f = 36

	iters = 10000

	if run == "opencl":
		a = np.fromfile('./weights/init_opencl_sofm_weights.dat', dtype=np.float32)
	else:
		a = np.fromfile('./weights/init_sofm_weights.dat', dtype=np.float32)

	a = a.reshape((network_dimensions[0] * network_dimensions[1], n_f))

	if run == "opencl":
		b = np.fromfile('./weights/final_opencl_sofm_weights.dat', dtype=np.float32)
	else:
		b = np.fromfile('./weights/final_sofm_weights.dat', dtype=np.float32)

	b = b.reshape(network_dimensions[0] * network_dimensions[1], n_f)


	if n_f == 3:

		print(a[-1,:])

		fig = plt.figure()
		# setup axes
		a = a.reshape(network_dimensions[0],network_dimensions[1], n_f)
		ax = fig.add_subplot(121, aspect='equal')
		ax.set_xlim((0, a.shape[0]+1))
		ax.set_ylim((0, a.shape[1]+1))
		ax.set_title('Self-Organising Map before')

		# plot the rectangles
		for x in range(1, a.shape[0] + 1):
			for y in range(1, a.shape[1] + 1):
				ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
							facecolor=a[x-1,y-1,:],
							edgecolor='none'))

		# setup axes
		ax = fig.add_subplot(122, aspect='equal')
		ax.set_xlim((0, b.shape[0]+1))
		ax.set_ylim((0, b.shape[1]+1))
		ax.set_title('Self-Organising Map after %d iterations' % iters)

		
		# plot the rectangles
		for x in range(1, b.shape[0] + 1):
			for y in range(1, b.shape[1] + 1):
				ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
							facecolor=b[x-1,y-1,:],
							edgecolor='none'))

	plt.show()

	activate = 1

	if activate and n_f > 3:

		print('loading genetic data...')
		gtex_gct_flt = np.load('../SciDAS/DeepGTEx/data/float_data/gtex_gct_data_float_v7.npy')
		total_gene_list = np.load('../SciDAS/DeepGTEx/data/gene_lists/gtex_gene_list_v7.npy')
		print('done')

		data = load_data('../SciDAS/DeepGTEx/data/class_counts/gtex_tissue_count_v7.json', gtex_gct_flt)

		s = 'HALLMARK_MYC_TARGETS_V2'

		subsets = read_subset_file('../SciDAS/DeepGTEx/subsets/hallmark_experiments.txt')
		for s in subsets:
			genes = []
			for g in subsets[s]:
				if g in total_gene_list:
					genes.append(g)
			subsets[s] = genes

		try:
			genes = subsets[s.upper()]
		except:
			print('Set not found in subset file, try again')
			sys.exit(1)

		gtex = GTEx(data, total_gene_list, genes)

		# get ordered class names
		plt.clf()
		classes = []
		for k in sorted(data.keys()):
			classes.append(str(k))

		# gather test data
		test_data = gtex.test.data.transpose()
		test_labels = gtex.test.labels.transpose()

		bmus_is = {}
		for i in range(test_data.shape[1]):
			idx = find_bmu(b, test_data[:,i].reshape(np.array([n_f, 1])), network_dimensions)
			label_num = np.argmax(test_labels[:,i])

			if classes[label_num] in bmus_is:
				bmus_is[classes[label_num]].append(idx)
			else:
				bmus_is[classes[label_num]] = []
				bmus_is[classes[label_num]].append(idx)

		x = []
		y = []   
		for b in sorted(bmus_is.keys()):    
			idxs = bmus_is[b]
			x_sum = 0
			y_sum = 0

			for i in idxs:
				x_sum = x_sum + i[0]
				y_sum = y_sum + i[1]

			x.append(x_sum / len(bmus_is[b]))
			y.append(y_sum / len(bmus_is[b]))

		fig = plt.figure()

		# for b in sorted(bmus_is.keys()):
		# 	for i in bmus_is[b]:
		# 		plt.scatter(i[0], i[1], marker='o')
		plt.scatter(x, y, marker='o')

		for label, i, j in zip(sorted(bmus_is.keys()), x, y):
			plt.annotate(label.split(' ')[0], (i, j))

		plt.show()







