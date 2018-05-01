#
# main.py
# run the sofm 
#

# essential libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from PIL import Image
import re
import json
import sys

# data and network imports
# from tensorflow.examples.tutorials.mnist import input_data
from sofm import sofm
from GTEx import GTEx

# read a csv or txt file that contains a name of a subset followed by a list of genes
def read_subset_file(file):
	with open(file, 'r') as f:
		content = f.readlines()

	# eliminate new line characters
	content = [x.strip() for x in content]

	# split on tabs or commas to create a sublist of set names and genes
	content = [re.split('\t|,', x) for x in content]

	# create a dictionary with keys subset names and values list of genes
	subsets = {}
	for c in content:
		subsets[c[0]] = c[1:]

	return subsets


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


# scales input for making the image
def scale(X, eps=1e-8):
	return (X - X.min())/ (X.max() + eps)


# this function allows for creating an image that displays each 28x28 neuron 
def make_tile(X, img_shape, tile_shape, tile_spacing=(2, 2)):
	out_shape = [(i + k) * j - k for i, j, k in zip(img_shape, tile_shape, tile_spacing)]

	H, W = img_shape
	Hs, Ws = tile_spacing

	out_array = np.ones(out_shape, dtype='uint8')
	out_array = out_array * 255

	for tile_row in xrange(tile_shape[0]):
		for tile_col in xrange(tile_shape[1]):

			if tile_row * tile_shape[1] + tile_col < X.shape[0]:
				img = scale(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
				out_array[
				tile_row * (H+Hs): tile_row * (H + Hs) + H,
				tile_col * (W+Ws): tile_col * (W + Ws) + W
				] = img * 255

	return out_array



if __name__ == '__main__':

	# load mnist data 
	# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	print('loading genetic data...')
	gtex_gct_flt = np.load('./data/gtex_gct_data_float_v7.npy')
	total_gene_list = np.load('./data/gtex_gene_list_v7.npy')
	print('done')

	data = load_data('./data/gtex_tissue_count_v7.json', gtex_gct_flt)

	s = 'HALLMARK_MYC_TARGETS_V2'

	subsets = read_subset_file('./data/hallmark_experiments.txt')
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

	data_in = gtex.train.data.transpose()

	print data_in.shape

	#np.save('./myc_targets_v2_train_data.dat', data_in)
	data_in.astype(np.float32).tofile('./myc_targets_v2_train_data.dat')

	sys.exit(0)

	# load random color data
	# raw_data = np.random.randint(0, 255, (3, 200))
	# data = raw_data.astype(np.float32) / raw_data.max()


	# define number of features
	n_f = len(genes)


	# define SOFM network dimensions
	network_dimensions = np.array([40, 40])


	# instantiate the network
	net = sofm(network_dimensions, 0.1, 15000, n_f)


	# collect the initial weights for visualization purposes
	a = np.array(net.weights, copy=True)


	# perform a train step for each iteration
	for i in range(net.iters):
		# extract input vector at random
		x = data_in[:, np.random.randint(0, data_in.shape[1])].reshape(np.array([n_f, 1]))

		# extract input vector using mnist api
		# x, y = mnist.train.next_batch(1)
		# x = x.reshape(n_f, 1) 
		# x = x.astype(np.float32) / 255.0

		net.train_step(x, i)
		if i % 100 == 0:
			print (str(net.lr) + '\t' + str(net.radius))


	# if n_f == len(genes):
	# 	test_data = gtex.test.data.transpose()
	# 	test_labels = gtex.test.labels.transpose()
	# 	bmus = []
	# 	for i in range(test_data.shape[1]):
	# 		bmus.append(net.find_bmu(test_data[:,i].reshape(np.array([n_f, 1]))))
	# 		print bmus[i], 

	activate = 1

	if activate:
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
			idx = net.find_bmu(test_data[:,i].reshape(np.array([n_f, 1])))
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





	if n_f == len(genes):
		fig = plt.figure()

		# setup axes
		dist_before = np.zeros((network_dimensions[0], network_dimensions[1]))
		a = a.reshape(network_dimensions[0],network_dimensions[1], n_f)

		for i in range(network_dimensions[0]): 
			for j in range(network_dimensions[1]):
				if i != j:
					dist_before[i,j] = dist_before[j,i] = np.sqrt(np.sum(np.power(a[i,j,:] - a[j,i,:], 2)))

		dist_before = dist_before / dist_before.max()


		ax = fig.add_subplot(121, aspect='equal')
		ax.set_xlim((0, a.shape[0]+1))
		ax.set_ylim((0, a.shape[1]+1))
		ax.set_title('Self-Organising Map before')

		# plot the rectangles
		for x in range(1, a.shape[0] + 1):
			for y in range(1, a.shape[1] + 1):
				ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
							facecolor=[dist_before[x-1,y-1],dist_before[x-1,y-1],dist_before[x-1,y-1]],
							edgecolor='none'))

		b = np.array(net.weights, copy=True)
		b = b.reshape(network_dimensions[0],network_dimensions[0],n_f)

		# setup axes
		dist_after = np.zeros((network_dimensions[0], network_dimensions[1]))

		for i in range(network_dimensions[0]): 
			for j in range(network_dimensions[1]):
				if i != j:
					dist_after[i,j] = dist_after[j,i] = np.sqrt(np.sum(np.power(b[i,j,:] - b[j,i,:], 2)))

		dist_after = dist_after / dist_after.max()

		# setup axes
		ax = fig.add_subplot(122, aspect='equal')
		ax.set_xlim((0, b.shape[0]+1))
		ax.set_ylim((0, b.shape[1]+1))
		ax.set_title('Self-Organising Map after %d iterations' % net.iters)

		# plot the rectangles
		for x in range(1, b.shape[0] + 1):
			for y in range(1, b.shape[1] + 1):
				ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
							color=[dist_after[x-1,y-1],dist_after[x-1,y-1],dist_after[x-1,y-1]],
							#facecolor=dist_after[x-1,y-1],
							edgecolor='none'))

		plt.show()


	# MNIST visualization
	if n_f == 784:
		W_b = make_tile(a, img_shape=(28,28), tile_shape=(network_dimensions[0],network_dimensions[1]))
		W_f = make_tile(net.weights, img_shape=(28,28), tile_shape=(network_dimensions[0],network_dimensions[1]))
		img1 = Image.fromarray(W_b)
		img2 = Image.fromarray(W_f)
		img1.show()
		img2.show()


	# random color visualization
	if n_f == 3:
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

		b = np.array(net.weights, copy=True)
		b = b.reshape(network_dimensions[0],network_dimensions[0],n_f)

		# setup axes
		ax = fig.add_subplot(122, aspect='equal')
		ax.set_xlim((0, b.shape[0]+1))
		ax.set_ylim((0, b.shape[1]+1))
		ax.set_title('Self-Organising Map after %d iterations' % net.iters)

		
		# plot the rectangles
		for x in range(1, b.shape[0] + 1):
			for y in range(1, b.shape[1] + 1):
				ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
							facecolor=b[x-1,y-1,:],
							edgecolor='none'))

		plt.show()



