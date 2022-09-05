import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import load_iris, load_digits


# dedicated classes
# ---------------------------------------------------------
class CategoricalClassifierNet:
	def __init__(self, data, targets, hiddenNeurons):
		assert data.shape[0] == targets.shape[0]
		self.rng = np.random.default_rng()
		self.X = data
		self.t = targets
		self.K = hiddenNeurons
		self.W1 = None
		self.W2 = None
		self.prior_update = None

	def __initWeights(self):
		self.W1 = np.vstack((np.zeros([1, self.X.shape[1]]) ,self.rng.standard_normal([self.K, self.X.shape[1]])))
		self.W2 = self.rng.standard_normal([self.t.shape[1], self.K+1])

	def __batch(self, epochs, batchsize):
		indices = np.arange(self.X.shape[0])
		batch = []
		start_epoch = True

		for epoch in range(epochs):
			# shuffle in  every new epoch
			self.rng.shuffle(indices)
			for index in indices:
				batch.append(index)
				# idea: carry partially filled batch into the new epoch
				if len(batch) == batchsize:
					yield self.X[batch], self.t[batch], start_epoch
					batch.clear()
					start_epoch = False
			start_epoch = True

		# yield last batch if not empty
		if(batch):
			yield self.X[batch], self.T[batch], True

	# auxiliary function used by the function __computeAccuracy() in order to encode the predictions
	# note: function is applied to a 1D-array (axis-wise)
	def __encode(self, predictions):
		# find most likely prediction
		p_max = np.amax(predictions)
		idx = np.where(predictions == p_max)[0]

		# Handling edge case with >1 max value
		if len(idx) > 1:
			# pick predicted class uniformely at random
			idx = np.array([self.rng.choice(idx)])
			pred_enc = np.zeros([len(predictions)])
			pred_enc[idx] = 1.
			return pred_enc			

		# encode predictions for non-edge cases
		return np.where(predictions == p_max, 1., 0.)

	def __sigmoid(self, exponent):
		return 1.0 / (1.0 + np.exp(-exponent))

	def __softMax(self, logits):
		Z_exp = np.exp(logits)
		return Z_exp / np.sum(Z_exp, axis=0)

	def __forward(self, data):
		H = self.__sigmoid(self.W1 @ np.transpose(data))
		H[0,:] = 1.0
		Z = self.W2 @ H
		Y = self.__softMax(Z)
		return Y, Z, H

	def __computeLoss(self, logits, targets):
		z_targets = logits[np.where(targets == 1.)[1] , np.arange(logits.shape[1])]
		# Note: Added mean for comparability in case of batch gradient descent
		#return (1./targets.shape[0]) * (-np.sum(z_targets - np.log(np.sum(np.exp(logits), axis=0))))
		return -np.sum(z_targets - np.log(np.sum(np.exp(logits), axis=0)))

	def __computeAccuracy(self):
		predictions,_,_ = self.__forward(self.X)
		predictions_enc = np.apply_along_axis(self.__encode, 0, predictions)
		matches = np.sum(np.logical_and(predictions_enc, np.transpose(self.t)), axis=0)
		acc = np.sum(matches) / len(matches)
		return acc

	def __gradient(self, data, predictions, targets, activations):
		grad_W1 = (1./data.shape[0]) * (((np.transpose(self.W2) @ (predictions - np.transpose(targets))) * activations * (1.-activations)) @ data)
		grad_W2 = (1./data.shape[0]) * ((predictions-np.transpose(targets)) @ np.transpose(activations))
		return grad_W1, grad_W2

	def __descent(self, data, targets, learning_rate, mu=None):
		Y,Z,H = self.__forward(data)
		loss = self.__computeLoss(Z, targets)
		# accuracy always computed on the entire dataset
		accuracy = self.__computeAccuracy()
		G1,G2 = self.__gradient(data, Y, targets, H)

		# descent step that is independent of the value of mu
		self.W1 -= learning_rate * G1
		self.W2 -= learning_rate * G2

		# Momentum learning: additional descent step if mu is not None
		if mu is not None:
			if self.prior_update is not None:
				self.W1 += mu * self.prior_update[0]
				self.W2 += mu * self.prior_update[1]
			self.prior_update = [-learning_rate*G1, -learning_rate*G2]

		return loss, accuracy

	def sgd(self, learning_rate=0.001, epochs=10000, batchsize=64, mu=None):
		print(f"Performing SGD with {epochs} epochs - eta={learning_rate}, B={batchsize}, mu={mu}")
		self.__initWeights()
		losses = []
		accuracies = []

		for iteration, (x,t,f) in enumerate(self.__batch(epochs, batchsize)):
			loss, accuracy = self.__descent(x, t, learning_rate, mu)
			if f is True:
				losses.append(loss)
				accuracies.append(accuracy)
			print("\rIteration: ", iteration, " - loss: ", loss, " - accuracy: ", float("{:.2f}".format(accuracy)), end="", flush=True)

		return np.array(losses), np.array(accuracies)
		

class OneHotEncoder:
	def encode(self, targets):
		t_in = targets.copy()
		t_enc = np.zeros([t_in.shape[0], np.unique(t_in).shape[0]])
		t_enc[np.arange(len(t_in)),t_in] = 1.
		return t_enc

		
# preparing datasets
# ---------------------------------------------------------
data_iris = load_iris()
data_digits = load_digits()
OneHotEncoder = OneHotEncoder()

X_iris = np.hstack((np.ones([data_iris.data.shape[0], 1]), data_iris.data))
t_iris_orig = data_iris.target
t_iris_enc = OneHotEncoder.encode(t_iris_orig)

X_digits = np.hstack((np.ones([data_digits.data.shape[0], 1]), data_digits.data))
t_digits_orig = data_digits.target
t_digits_enc = OneHotEncoder.encode(t_digits_orig)




# main
# ---------------------------------------------------------
datasets = 1
K = [150]
learning_rate = [0.001]
epochs = [100000]
batchsize = [X_iris.shape[0]]
mu = [0.5]
losses = []
accuracies = []

cat_clf_iris = CategoricalClassifierNet(X_iris, t_iris_enc, K[0])
loss, accuracy = cat_clf_iris.sgd(learning_rate=learning_rate[0], epochs=epochs[0], batchsize=batchsize[0], mu=mu[0])
losses.append(loss)
accuracies.append(accuracy)

pdf = PdfPages("Categorical_Classifier_Net.pdf")

# plots
for dataset in range(datasets):
	fig, ax = plt.subplots()
	ax.plot(np.arange(losses[dataset].shape[0]), losses[dataset], '-', color="blue")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.set_xscale("log")

	ax2 = ax.twinx()
	ax2.plot(np.arange(accuracies[dataset].shape[0]), accuracies[dataset], '-', color="red")
	ax2.set_ylabel("Accuracy")

	pdf.savefig()
	plt.show()

pdf.close()





