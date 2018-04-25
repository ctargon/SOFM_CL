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


if __name__ == '__main__':

	network_dimensions = [20,20]
	n_f = 3

	iters = 5000

	a = np.fromfile('./weights/init_opencl_sofm_weights.dat', dtype=np.float32)

	a = a.reshape((network_dimensions[0] * network_dimensions[1], n_f))

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


	b = np.fromfile('./weights/final_opencl_sofm_weights.dat', dtype=np.float32)
	b = b.reshape((network_dimensions[0] * network_dimensions[1], n_f))

	print b.shape

	print(b[-1,:])

	b = b.reshape(network_dimensions[0],network_dimensions[1], n_f)

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









