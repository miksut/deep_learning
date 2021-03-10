import numpy as np
import math

# dedicated class and functions according to task descrition
class GradientDescent:
	# weight input "w_init" must be a numpy 2d-array and the learning rate a 1d-array
	def __init__(self, l_func, grad, w_init, lr):
		self.loss_function = l_func
		self.gradient = grad
		self.weigths_init = w_init
		self.learning_rate = lr

	def run_gd(self):
		#results = np.
		for w in range(self.weigths_init.shape[0]):
			for lr in range(self.learning.shape[0]):


				pass




		loss = math.inf
		w = self.w_init
		



# Loss function: J = (w_1)² + (w_2)² + 30sin(w_1)sin(w_2)
def loss_function(w):
	return np.power(w[0],2) + np.power(w[1], 2) + 30 * np.sin(w[0]) * np.sin(w[1])

def gradient():
	print("This is the gradient")







#obj = GradientDescent(loss_function, gradient)

w_init = np.array([1,2])

dt = np.dtype([('name', np.unicode_, 16), ('numbers', np.float64, (2,))])
tup = ("Test", (1.5, 2.5))

arr = np.ones((2,3), dtype=dt)
arr[0][0] = tup
print(arr[0][0])
print(arr[0][0]['numbers'])
print(arr.shape)
