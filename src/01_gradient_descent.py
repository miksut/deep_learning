import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.backends.backend_pdf import PdfPages


# global variables
iterations = 5000
rng = np.random.default_rng()

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
				w_current = self.weights_init[w].copy()
				loss = self.loss_function(w_current)

				# stopping criterion part 1: limit the number of iterations
				for j in range(iterations):
					grad = self.gradient(w_current)

					# stopping criterion part 2: if norm of gradient is small -> stop
					if scipy.linalg.norm(grad) < 1e-4:
						break

					w_current -= self.learning_rate[lr] * grad
					loss = self.loss_function(w_current)

				results[w][lr]['loss'] = loss
				results[w][lr]['weights'] = w_current

		return results


# Loss function: J = (w_1)² + (w_2)² + 30sin(w_1)sin(w_2)
def loss_function(w):
	return np.power(w[0],2) + np.power(w[1],2) + (30*np.sin(w[0])*np.sin(w[1]))

def gradient(w):
	# Note: an elegant alternative for implementation is to return 2.0*w + 30.0 * np.cos(w) * np.sin(w[::-1]), i.e. flipping the order of the elements in the last term
	grad = np.zeros((w.shape[0]))
	grad[0] = (2*w[0]) + (30*np.cos(w[0])*np.sin(w[1]))
	grad[1] = (2*w[1]) + (30*np.sin(w[0])*np.cos(w[1]))
	return grad


# main
# randomly initialize arguments in a given interval
weights_init = rng.uniform(-10.0, 10.0, (5,2))
learning_rate = np.array([1.0, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001])

gd = GradientDescent(loss_function, gradient, weights_init, learning_rate)
res = gd.run_gd()

# evaluation results
print(res)

# Plotting
# save evaluation results in a pdf file
pdf = PdfPages("../results/task01_evaluation.pdf")

for i in range(weights_init.shape[0]):
	plt.plot(learning_rate, res[i]['loss'])
	plt.plot(learning_rate, res[i]['loss'], 'o')
	plt.title(f"Initialised Weights: {weights_init[i]}")
	plt.xlabel("Learning Rate")
	plt.ylabel("Optimised Loss")
	plt.yscale("symlog")
	plt.xticks(learning_rate)
	pdf.savefig()
	plt.close()
pdf.close()

# plot error surface and best result
x = np.arange(-10.0, 10.0, 0.1)
y = np.arange(-10.0, 10.0, 0.1)
xx, yy = np.meshgrid(x,y)
zz = loss_function([xx,yy])

# optimal solution of the evaluation procedure
idx_w = np.where(res[:][:]['loss'] == np.amin(res[:][:]['loss']))[0][0]
idx_lr = np.where(res[:][:]['loss'] == np.amin(res[:][:]['loss']))[1][0]

loss_opt = res[idx_w][idx_lr]['loss']
weights_opt = res[idx_w][idx_lr]['weights']
weights_start = weights_init[idx_w]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', azim=-40, elev=50)
ax.plot_surface(xx, yy, zz, cmap='jet', alpha=0.4)
ax.plot(weights_start[0], weights_start[1], loss_function(weights_start), 'ro')
ax.plot(weights_opt[0],  weights_opt[1], loss_opt, 'go')
plt.show()
pdf.close()


