import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# dedicated class
class BinaryClassifier2LayerNet:
	def __init__(self, data, targets, hidden_neurons):
		assert data.shape[0] == targets.shape[0]
		self.rng = np.random.default_rng()
		self.X = data
		self.t = targets
		self.K = hidden_neurons
		self.W1 = None
		self.W2 = None

	def __initWeights(self):
		self.W1 = np.vstack((np.zeros([1, self.X.shape[1]]), self.rng.standard_normal([self.K, self.X.shape[1]])))
		self.W2 = self.rng.standard_normal(self.K+1)

	def __batch(self):
		pass

	def __sigmoid(self):
		pass

	def __forward(self):
		pass

	def __computeLoss(self):
		pass

	def __gradient(self):
		pass

	def __descent(self):
		pass

	def sgd(self):
		self.__initWeights()

		


# preparing datasets
# ---------------------------------------------------------
df_banknotes = pd.read_csv("datasets/data_banknote_authentication.txt", sep=',', names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'])
df_spam = pd.read_csv("datasets/spambase/spambase.data", sep=',', names=np.arange(1,59))

X_bank = np.hstack((np.ones([df_banknotes.shape[0], 1]), df_banknotes.drop(['Label'], axis=1).values))
t_bank = df_banknotes.iloc[:,-1].values

X_spam = np.hstack((np.ones([df_spam.shape[0], 1]), df_spam.drop([58], axis=1).values))
t_spam = df_spam.iloc[:,-1].values
