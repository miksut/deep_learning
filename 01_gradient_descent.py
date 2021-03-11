import numpy as np
import math
import matplotlib.pyplot as plt


# dedicated class and functions according to task descrition
class GradientDescent:
	# weight input "w_init" must be a numpy 2d-array and the learning rate a 1d-array
	def __init__(self, l_func, grad, w_init, lr):
		self.loss_function = l_func
		self.gradient = grad
		self.weights_init = w_init
		self.learning_rate = lr

	def run_gd(self):
		dt = np.dtype([('loss', np.float64),('weights', np.float64, (2,))])
		results = np.zeros((self.weights_init.shape[0], self.learning_rate.shape[0]), dtype=dt)

		for w in range(self.weights_init.shape[0]):
			for lr in range(self.learning_rate.shape[0]):
				loss = math.inf
				w_t0 = self.weights_init[w]
				w_t1 = self.weights_init[w]

				while (loss - self.loss_function(w_t1)) >= 1.0:
					loss = self.loss_function(w_t1)
					w_t0 = w_t1

					grad = self.gradient(w_t1)
					w_t1 = w_t1 - self.learning_rate[lr] * grad

				results[w][lr]['loss'] = loss
				results[w][lr]['weights'] = w_t0

		return results

# Loss function: J = (w_1)² + (w_2)² + 30sin(w_1)sin(w_2)
def loss_function(w):
	return np.power(w[0],2) + np.power(w[1],2) + (30*np.sin(w[0])*np.sin(w[1]))

def gradient(w):
	grad = np.zeros((w.shape[0]))
	grad[0] = (2*w[0]) + (30*np.cos(w[0])*np.sin(w[1]))
	grad[1] = (2*w[1]) + (30*np.sin(w[0])*np.cos(w[1]))
	return grad


# main
weights_init = np.array([[8.0, 8.0], [5.0, 5.0], [2.5, 2.5], [0.0, 0.0], [-7.5, -7.5]])
learning_rate = np.array([1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

gd = GradientDescent(loss_function, gradient, weights_init, learning_rate)
res = gd.run_gd()
print(res)
print(res[0][:]['loss'])


# Plotting
fig_1 = plt.figure()

for i in range(weights_init.shape[0]):
	ax_1 = fig_1.add_subplot(3,2,(i+1), title=f"Initialised Weights: {weights_init[i]}", xlabel="Learning Rate", ylabel="Optimised Loss" )
	ax_1.plot(learning_rate, res[i]['loss'])
	ax_1.plot(learning_rate, res[i]['loss'], 'o')

plt.subplots_adjust(hspace=0.3)
plt.show()


x = np.arange(-10.0, 10.0, 0.1)
y = np.arange(-10.0, 10.0, 0.1)
xx, yy = np.meshgrid(x,y)
zz = loss_function([xx,yy])

fig_2 = plt.figure()
ax_21 = fig_2.add_subplot(111, projection='3d', azim=-40, elev=50)
ax_21.plot_surface(xx, yy, zz, cmap='jet', alpha=0.4)
ax_21.plot(weights_init[-1][0], weights_init[-1][1], loss_function(weights_init[-1]), 'ro')
ax_21.plot(res[-1][4]['weights'][0], res[-1][4]['weights'][1], res[-1][4]['loss'], 'go')
plt.show()
plt.savefig("Surface.pdf")