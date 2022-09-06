import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# global variables and settings
rng = np.random.default_rng(20)

# dedicated class and functions
class TwoLayerNet:
	def __init__(self, data, hiddenNodes):
		self.X = data
		self.D = self.X.shape[1] - 2
		self.K = hiddenNodes
		self.W_1 = np.vstack((np.zeros([1, self.D+1]) ,rng.standard_normal([self.K, self.D+1])))
		self.w_2 = rng.standard_normal(self.K+1)
		self.h = np.zeros([self.K+1, self.X.shape[0]])
		self.y = np.zeros([self.X.shape[0], 1])


	def __sigmoid(self, a):
		return 1.0 / (1.0 + np.exp(-a))


	def __forwardPass(self):
		#print(f"Current weights: W_1 = {self.W_1} and w_2 = {self.w_2}")
		self.h = self.__sigmoid((self.W_1 @ np.transpose(self.X[:,:-1])))
		self.h[0,:] = 1.0
		self.y = np.transpose(self.w_2) @ self.h


	# computation of gradient according to lecture notes
	def __computeGradientW1(self):
		return (2.0 / self.X.shape[0]) * ((np.outer(self.w_2, (self.y - self.X[:,-1])) * self.h * (1-self.h)) @ self.X[:,:-1])

		
	# computation of gradient according to lecture notes
	def __computeGradientw2(self):
		return 2*np.mean((self.y - self.X[:,-1]).flatten() * self.h, axis=1) 


	def __computeLoss(self):
		return (1.0 / self.X.shape[0]) * np.sum(np.power((self.y - self.X[:,-1]), 2))


	def run_gd(self, learning_rate, epochs):
		loss = np.zeros(epochs)

		for epoch in range(epochs):
			# perform forward pass
			self.__forwardPass()
			loss[epoch] = self.__computeLoss()

			# computing gradients
			grad_W_1 = self.__computeGradientW1()
			grad_w_2 = self.__computeGradientw2()

			# adjusting the weight vectors
			self.W_1 = self.W_1 - learning_rate * grad_W_1
			self.w_2 = self.w_2 - learning_rate * grad_w_2

		return loss


	# apply the current model onto new data
	def applyModel(self, data):
		X = data.copy()
		h = self.__sigmoid((self.W_1 @ np.transpose(X[:,:-1])))
		h[0,:] = 1.0
		X[:,-1] = np.transpose(self.w_2) @ h

		return X


# class for finding appropriate parameters
class ParamOptimiser:
	def __init__(self, data, hidden_neurons, learning_rate, epochs, low, high):
		self.X = data
		self.K = hidden_neurons
		self.eta = learning_rate
		self.epochs = epochs
		self.dataCreator = DatasetCreation()
		self.low = low
		self.high = high


	def optimise(self, name):
		filename = name
		pdf = PdfPages("../results/" + name + ".pdf")
		X_plot = self.dataCreator.plotData(self.low, self.high, 1000)


		for i in range(self.K.shape[0]):
			for j in range(self.eta.shape[0]):
				fig = plt.figure()
				for k in range(self.epochs.shape[0]):
					net = TwoLayerNet(self.X, self.K[i])
					loss = net.run_gd(self.eta[j], self.epochs[k])

					# plots
					X_plot = net.applyModel(X_plot)

					ax_1 = fig.add_subplot(self.epochs.shape[0],2,(k+1)+k)
					ax_1.plot(self.X[:,1], self.X[:,-1], 'rx', label=f"K={self.K[i]}, eta={self.eta[j]}, epochs={self.epochs[k]}")
					ax_1.plot(X_plot[:,1], X_plot[:,-1], '-')
					ax_1.legend(loc='upper right', fontsize='x-small')

					ax_2 = fig.add_subplot(self.epochs.shape[0],2,2*(k+1))
					ax_2.plot(np.linspace(0, self.epochs[k], self.epochs[k]), loss, label=f"Min. loss={np.amin(loss):4f}")
					ax_2.legend(loc='upper right', fontsize='x-small', markerscale=0.5)
					print("Check")

				pdf.savefig()
				plt.show()
				plt.close()
		pdf.close()


# functions for coputing targets according to task description
class DatasetCreation:
	def cosine(self, samples):
		X = np.ones([samples, 3])
		X[:,1] = rng.uniform(-2, 2, samples)
		X[:,-1] = ((np.cos(3*X[:,1]) + 1) * 0.5)
		return X

	def gaussian(self, samples):
		X = np.ones([samples, 3])
		X[:,1] = rng.uniform(-2,2, samples)
		X[:,-1] = (np.exp((np.power(X[:,1],2) * -0.25)))
		return X

	def polynomial(self, samples):
		X = np.ones([samples, 3])
		X[:,1] = rng.uniform(-4.5, 3.5, samples)
		poly = (np.polynomial.Polynomial([64, 10, 27, 11, 3, 1]))
		X[:,-1] = 0.01 * poly(X[:,1])
		return X

	def plotData(self, low, high, samples):
		X = np.ones([samples, 3])
		X[:,1] = np.linspace(low, high, samples)
		return X


# main
dataCreator = DatasetCreation()


"""
# Evaluation part: find appropriate parameters (eta, learning rate, epochs) for the network
samples = np.array([50, 1000])
K = np.array([1, 2, 5, 10, 20, 50])
lr = np.array([0.5, 0.3, 0.2, 0.1, 0.05])
epochs = np.array([1, 10, 100, 1000, 5000, 10000])

for i in range(samples.shape[0]):
	# evaluation procedure for cosine function
	X_cos = dataCreator.cosine(samples[i])
	cosOpt = ParamOptimiser(X_cos, K, lr, epochs, -2, 2)
	cosOpt.optimise(f"Cosine_{samples[i]}")

	# evaluation procedure for Gaussian function
	X_gauss = dataCreator.gaussian(samples[i])
	gaussOpt = ParamOptimiser(X_gauss, K, lr, epochs, -2, 2)
	gaussOpt.optimise(f"Gauss_{samples[i]}")

	# evaluation procedure for polynomial function
	X_poly = dataCreator.polynomial(samples[i])
	polyOpt = ParamOptimiser(X_poly, K, lr, epochs, -4.5, 3.5)
	polyOpt.optimise(f"Poly_{samples[i]}")
"""

# based on the evaluation (see outcommented code above), the network should work well with: K=5, eta=0.1, epochs=10'000
samples = np.array([50])
K = np.array([5])
lr = np.array([0.1])
epochs = np.array([10000])

X_cos = dataCreator.cosine(samples[0])
cosOpt = ParamOptimiser(X_cos, K, lr, epochs, -2, 2)
cosOpt.optimise("Cosine_opt")

X_gauss = dataCreator.gaussian(samples[0])
gaussOpt = ParamOptimiser(X_gauss, K, lr, epochs, -2, 2)
gaussOpt.optimise("Gauss_opt")

X_poly = dataCreator.polynomial(samples[0])
polyOpt = ParamOptimiser(X_poly, K, lr, epochs, -4.5, 3.5)
polyOpt.optimise("Poly_opt")