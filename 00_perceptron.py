# import
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# global variables
dataset_size = 20
epochs = 5
learning_rate = 1

# dataset creation
mean = [[0,0], [15, 15]]
cov = [[[5,0],[0,5]], [[5,0],[0,5]]]
labels = [np.ones((dataset_size,1))*-1, np.ones((dataset_size,1))]
rng = np.random.default_rng(20)
data_1 = np.append(rng.multivariate_normal(mean[0], cov[0], dataset_size), labels[0], axis=1)
data_2 = np.append(rng.multivariate_normal(mean[1], cov[1], dataset_size), labels[1], axis=1)
data = np.append(data_1, data_2, axis=0)
data = np.c_[np.ones((2*dataset_size,1)), data]

scaler = preprocessing.StandardScaler().fit(data)
#data_scaled = scaler.transform(data)
rng.shuffle(data)

# perceptron algorithm and auxiliary function
def predict(row, weights):
	activation = np.dot(row, weights)
	return 1.0 if activation >= 0 else -1.0

def perceptron_learning(X_data, y_data, w_init, n_epochs, l_rate):
	weights = w_init
	for epoch in range(n_epochs):
		for row in range(X_data.shape[0]):
			if np.dot(predict(X_data[row,:], weights), y_data[row]) < 0:
				weights = weights + l_rate * y_data[row] * X_data[row,:]
				print(f"Row: {row}\t Updated weights: {weights}")
	return weights

def plot_boundary(X_data, weights):
	y = np.zeros(X_data.shape[0])
	slope = -(weights[1]/weights[2])
	intercept = -(weights[0]/weights[2])
	for point in range(X_data.shape[0]):
		y[point] = slope * X_data[point] + intercept
	return y

# main
weights_init = rng.standard_normal(data.shape[1]-1)
weights = perceptron_learning(data[:,:-1], data[:,-1], weights_init, epochs, learning_rate)

# plotting
boundary_x = np.arange(-10,25,0.01)
boundary_y = plot_boundary(boundary_x, weights)

plt.plot(data_1[:,0], data_1[:,1], 'x')
plt.plot(data_2[:,0], data_2[:,1], 'o')
plt.plot(boundary_x, boundary_y, '-')
plt.axis('equal')
plt.show()