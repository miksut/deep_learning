import numpy as np
import matplotlib.pyplot as plt
import math

# global variables and settings
dataset_size = 100
seed = 30
rng = np.random.default_rng(seed)
coef_a = 1.5
bias = -3
noise = rng.standard_normal(dataset_size) * 1.25

# dedicated class and functions according to task description
class LinearRegressionGD:
	def __init__(self, X, w, model, loss_fct, gradient):
		self.data = X
		self.weights_init = w
		self.model = model
		self.loss_function = loss_fct
		self.gradient = gradient
		self.learning_rate = 0.0003

	def run_gd(self):
		loss = math.inf
		w_t0 = self.weights_init
		w_t1 = self.weights_init

		while (loss - self.loss_function(self.data, w_t1, self.model)) >= 1.0:
			print("LOSS BEFORE")
			print(loss)

			loss = self.loss_function(self.data, w_t1, self.model)
			w_t0 = w_t1

			print("LOSS AFTER")
			print(loss)

			grad = self.gradient(self.data, self.weights_init, self.model)
			w_t1 = w_t1 - self.learning_rate * grad
			print("Entered")
			print(w_t1)
			print("LOSS NEW")
			print(self.loss_function(self.data, w_t1, self.model))


		print("Exit")
		print(w_t0)




# w needs to be a 1d np-array
def linear_unit(X, w):
	return np.matmul(X,w)

def loss_function(X, w, model):
	size = X.shape[0]
	return (1/size) * np.sum(np.power((model(X[:,:-1], w) - X[:,-1]), 2))

def gradient(X, w, model):
	grad = np.zeros((w.shape[0]))
	size = X.shape[0]
	grad[0] = (1/size) * np.sum(model(X[:,:-1], w) - X[:,-1])
	grad[1] = (1/size) * (np.sum((model(X[:,:-1], w) - X[:,-1]) * X[:,1]))
	return grad


# dataset
data = np.zeros([dataset_size,3])
data[:,0] = np.ones(dataset_size)
data[:,1] = np.abs(rng.standard_normal(dataset_size)*10)
data[:,2] = coef_a * data[:,1] + bias + noise


w = np.array([0.1, -0.2])

reg_gd = LinearRegressionGD(data, w, linear_unit, loss_function, gradient)
reg_gd.run_gd()

print("Gradient")
print((1/100) * (np.sum((linear_unit(data[:,:-1], w) - data[:,-1]) * data[:,1])))

print(data[:,:2])
print(np.matmul(data[:,:2], np.transpose(w))*np.matmul(data[:,:2], np.transpose(w)))

print(np.sum(np.power(linear_unit(data[:,:-1], w),2)))
print(loss_function(data, w, linear_unit))
print(w.shape)
print(gradient(data, w, linear_unit))


# plotting
x = np.arange(0,30,0.1)
y = coef_a * x + bias

plt.plot(data[:,1], data[:,2], 'x')
plt.plot(x, y, 'r')

#plt.show()
