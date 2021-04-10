import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# dedicated class
class BinaryClassifierNet:
	def __init__(self, data, targets, hidden_neurons):
		assert data.shape[0] == targets.shape[0]
		self.rng = np.random.default_rng()
		self.X = data
		self.t = targets
		self.K = hidden_neurons
		self.W1 = None
		self.W2 = None
		self.prior_update = None

	def __initWeights(self):
		self.W1 = np.vstack((np.zeros([1, self.X.shape[1]]), self.rng.standard_normal([self.K, self.X.shape[1]])))
		self.W2 = self.rng.standard_normal(self.K+1)

	def __batch(self, epochs, batchsize):
		indices = np.arange(self.X.shape[0])
		batch = []
		start_epoch = True

		for epoch in range(epochs):
			# shuffle in every new epoch
			self.rng.shuffle(indices)
			for index in indices:
				batch.append(index)
				# idea: carry partially filled batch into the new epoch
				if len(batch) == batchsize:
					yield self.X[batch], self.t[batch], start_epoch
					batch.clear()
					start_epoch = False
			start_epoch = True

		# yield the last batch if not emptoy
		if batch:
			yield self.X[batch], self.T[batch], True

	def __sigmoid(self, exponent):
		return 1.0 / (1.0 + np.exp(-exponent))

	def __softPlus(self, logits):
		return np.log(1+np.exp(logits))

	def __forward(self, data):
		H = self.__sigmoid(self.W1 @ np.transpose(data))
		H[0,:] = 1.0
		z = np.transpose(self.W2) @ H
		y = self.__sigmoid(z)
		return y, z, H

	def __computeLoss(self, logits, targets):
		return np.sum(targets*self.__softPlus(-logits) + (1.0-targets)*self.__softPlus(logits))

	def __accuracy(self):
		y,_,_ = self.__forward(self.X)
		t = self.t

		# If y_n < 0.5, set to 0, else set to 1
		y[y < 0.5] = 0
		y[y >= 0.5] = 1

		correct_classifications = np.logical_not(np.logical_xor(y, t))
		return np.sum(correct_classifications) / correct_classifications.shape[0]

	def __gradient(self, data, targets, predictions, activations):
		G1 = (np.outer(self.W2, (predictions-targets)) * activations * (1.0-activations)) @ data
		G2 = np.mean((predictions - targets).flatten() * activations, axis=1)
		return G1, G2

	def __descent(self, data, targets, learning_rate, mu=None):
		y,z,H = self.__forward(data)
		loss = self.__computeLoss(z, targets)
		# accuracy always computed on the entire dataset
		accuracy = self.__accuracy()
		G1,G2 = self.__gradient(data, targets, y, H)

		# descent step that is independent of the value of mu
		self.W1 -= learning_rate * G1
		self.W2 -= learning_rate * G2

		# Momentum Learning: Additional descent step if mu is not None
		if mu is not None:
			if self.prior_update is not None:
				self.W1 += mu * self.prior_update[0]
				self.W2 += mu * self.prior_update[1]
			self.prior_update = [-learning_rate*G1, -learning_rate*G2]

		return loss, accuracy

	def sgd(self, learning_rate=0.001, epochs=100000, batchsize=64, mu=None):
		print(f"Performing SGD with {epochs} epochs - eta={learning_rate}, B={batchsize}, mu={mu}")
		self.__initWeights()
		losses = []
		accuracies = []

		for iteration, (x,t,f) in enumerate(self.__batch(epochs, batchsize)):
			loss, accuracy = self.__descent(x, t, learning_rate, mu)
			if f == True:
				losses.append(loss)
				accuracies.append(accuracy)
			print("\rIteration: ", iteration, " - loss: ", loss, " - accuracy: ", float("{:.2f}".format(accuracy)), end="", flush=True)

		return np.array(losses), np.array(accuracies)


# preparing datasets
# ---------------------------------------------------------
df_banknotes = pd.read_csv("datasets/data_banknote_authentication.txt", sep=',', names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'])
df_spam = pd.read_csv("datasets/spambase/spambase.data", sep=',', names=np.arange(1,59))

X_bank = np.hstack((np.ones([df_banknotes.shape[0], 1]), df_banknotes.drop(['Label'], axis=1).values))
t_bank = df_banknotes.iloc[:,-1].values

X_spam = np.hstack((np.ones([df_spam.shape[0], 1]), df_spam.drop([58], axis=1).values))
t_spam = df_spam.iloc[:,-1].values


# main
# ---------------------------------------------------------
datasets = 1
K = 15
learning_rate = [0.001]
epochs = [1000]
batchsize = [X_bank.shape[0]]
losses = []
accuracies = []

bin_clf1 = BinaryClassifierNet(X_bank, t_bank, K)
loss, acc = bin_clf1.sgd(learning_rate=learning_rate[0], epochs=epochs[0], batchsize=batchsize[0])
losses.append(loss)
accuracies.append(acc)

pdf = PdfPages("Binary_Classifier_Net.pdf")

# plots
for dataset in range(datasets):
	fig,ax = plt.subplots()
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