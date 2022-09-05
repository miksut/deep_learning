# imports
import numpy as np
import matplotlib.pyplot as plt


# global variables and settings
dataset_size = 30
seed = 30
rng = np.random.default_rng(seed)


# creating 2d-dataset and random shuffling
mean = [[1,-5,-10], [1,5,10]]
cov = [[[0,0,0], [0,15,0], [0,0,10]], [[0,0,0], [0,20,0], [0,0,10]]]
labels = [np.ones((dataset_size,1))*-1, np.ones((dataset_size,1))]

data_neg = np.append(rng.multivariate_normal(mean[0], cov[0], dataset_size), labels[0], axis=1)
data_pos = np.append(rng.multivariate_normal(mean[1], cov[1], dataset_size), labels[1], axis=1)
data = np.append(data_neg, data_pos, axis=0)
rng.shuffle(data)


# perceptron algorithm and auxiliary function
def set_globals(data_size, rnd_seed):
	dataset_size = data_size
	seed = rnd_seed
	rng = np.random.default_rng(seed)

def predict(row, weights):
	activation = np.dot(row, weights)
	return 1.0 if activation >= 0 else -1.0

def perceptron_learning(X_data, y_data, w_init):
	missclassifications = X_data.shape[0]
	weights = w_init

	# stopping criterion: one epoch without any missclassification
	while missclassifications > 0:
		missclassifications = 0
		for row in range(X_data.shape[0]):
			if np.dot(predict(X_data[row,:], weights), y_data[row]) < 0:
				missclassifications += 1
				weights = weights + y_data[row] * X_data[row,:]
				print(f"Row: {row}\t Updated weights: {weights}")
	return weights

def plot_boundary(X_data, weights):
	y = np.zeros(X_data.shape[0])
	slope = -(weights[1]/weights[2])
	intercept = -(weights[0]/weights[2])
	for point in range(X_data.shape[0]):
		y[point] = slope * X_data[point] + intercept
	return y


# main logic and plotting
#set_globals(20,20)
for sz, sd in [(30, 20), (30, 120), (30, 220), (40, 320), (50, 420)]:
	set_globals(sz, sd)

	weights_init = rng.standard_normal(data.shape[1]-1)
	weights = perceptron_learning(data[:,:-1], data[:,-1], weights_init)

	boundary_x = np.arange(-20,20,0.01)
	boundary_y = plot_boundary(boundary_x, weights)

	plt.plot(data_neg[:,1], data_neg[:,2], 'x')
	plt.plot(data_pos[:,1], data_pos[:,2], 'o')
	plt.plot(boundary_x, boundary_y, '-')
	plt.axis('equal')
	plt.show()