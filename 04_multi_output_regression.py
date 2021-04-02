import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from matplotlib.backends.backend_pdf import PdfPages


# dedicated class
# ---------------------------------------------------------
# assumption: data matrices store samples row-wise and already include the bias feature
class MultiOutput2LN:
	def __init__(self, X_, T_, hiddenNeurons):
		assert X_.shape[0] == T.shape[0]
		self.rng = np.random.default_rng(100)
		self.X = X_
		self.T = T_
		self.K = hiddenNeurons
		self.W1 = None
		self.W2 = None
		self.prior_update = None

	def __initWeights(self):
		self.W1 = np.vstack((np.zeros([1, self.X.shape[1]]), self.rng.standard_normal([self.K, self.X.shape[1]])))
		self.W2 = self.rng.standard_normal([self.T.shape[1], self.K+1])

	def __batch(self, B, epochs):

		indices = np.arange(self.X.shape[0])
		batch = []
		start_epoch = True

		for epoch in range(epochs):
			self.rng.shuffle(indices)
			for index in indices:
				batch.append(index)
				# idea: carry partially filled batch into the new epoch
				if len(batch) == B:
					yield self.X[batch], self.T[batch], start_epoch
					batch.clear()
					start_epoch = False
			start_epoch = True

		# yield the last batch if not emptoy
		if batch:
			yield self.X[batch], self.T[batch], True

	def __sigmoid(self, A):
		return expit(A)
		#return 1.0/(1.0+np.exp(-A))


	def __forward(self, X):
		H = self.__sigmoid(self.W1 @ np.transpose(X))
		H[0,:] = 1.0
		Y = self.W2 @ H
		return H, Y

	def __computeLoss(self, X, Y, T):
		return (1.0/X.shape[0])*np.power(np.linalg.norm(Y - np.transpose(T)), 2)

	def __gradient(self, X, Y, T, H):
		grad_W1 = (2.0/X.shape[0]) * (((np.transpose(self.W2) @ (Y - np.transpose(T))) * H * (1-H)) @ X)
		grad_W2 = (2.0/X.shape[0]) * ((Y-np.transpose(T)) @ np.transpose(H))
		return grad_W1, grad_W2

	def __descent(self, X, T, eta, mu=None):
		H, Y = self.__forward(X)
		loss = self.__computeLoss(X, Y, T)
		G1, G2 = self.__gradient(X, Y, T, H)

		# descent step that is independent of the value of mu
		self.W1 -= eta * G1
		self.W2 -= eta * G2

		# Momentum Learning: Additional descent step if mu is not None
		if mu is not None:
			if self.prior_update is not None:
				self.W1 += mu * self.prior_update[0]
				self.W2 += mu * self.prior_update[1]
		self.prior_update = [-eta * G1, -eta * G2]

		return loss

	def gradient_descent(self, eta=0.001, mu=None, epochs=100000):
		print(f"Performing Gradient Descent for {epochs} epochs")
		self.__initWeights()
		losses = []

		for epoch in range(epochs):
			loss = self.__descent(self.X, self.T, eta, mu)
			print("\repoch: ", epoch+1, " - loss: ", loss, end="", flush=True)
			losses.append(loss)
		print("")

		return np.array(losses)

	def stochastic_gradient_descent(self, eta=0.001, mu=None, epochs=100000, batchsize=64):
		print(f"Performing Stochastic Gradient Descent for {epochs} epochs with batch size {batchsize}")
		self.__initWeights()
		losses = []

		for iteration, (x, t, f) in enumerate(self.__batch(batchsize, epochs)):
			loss = self.__descent(x, t, eta, mu)
			if f == True:
				losses.append(loss)
			print("\riteration: ", iteration, " - loss: ", loss, end="", flush=True)
		print("")

		return np.array(losses)

	def evaluate(self, X, index, name, values=[1,-1]):
		print("")
		print(f"Evaluating {name}")

		# considering the dropped categorical features
		if(index > 8): index -= 4

		for value in values:
			x = X[X[:,index] == value]
			_,y = self.__forward(x)
			mean = np.mean(y, axis=1)
			print(f"Average grades for {value} are {mean}")

		
# Preparing dataset (available at https://archive.ics.uci.edu/ml/datasets/Student+Performance#)
# ---------------------------------------------------------
df = pd.read_csv('student/student-mat.csv', sep=';')
# drop categorical values (according to task description)
df = df.drop(df.iloc[:, 8:12].columns, axis=1)
# change dtypes of columns according to task description
dict_float = {key: "float64" for key in df.select_dtypes(include=['int64']).columns}
df = df.astype(dict_float)
list_bin = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
[df[i].replace(to_replace=['no', 'yes'], value=[-1.0, 1.0], inplace=True) for i in list_bin]
df['school'].replace(to_replace=['GP', 'MS'], value=[1.0, -1.0], inplace=True)
df['sex'].replace(to_replace=['F', 'M'], value=[1.0, -1.0], inplace=True)
df['address'].replace(to_replace=['U', 'R'], value=[1.0, -1.0], inplace=True)
df['famsize'].replace(to_replace=['LE3', 'GT3'], value=[1.0, -1.0], inplace=True)
df['Pstatus'].replace(to_replace=['T', 'A'], value=[1.0, -1.0], inplace=True)

X = df.drop(['G1', 'G2', 'G3'], axis=1).values
# adding bias term
X= np.hstack((np.ones([X.shape[0], 1]), X))
T = df[['G1', 'G2', 'G3']].values


# main
# ---------------------------------------------------------
K = 75
learning_rate = 0.001
epochs = 10000
batchsize = 64

pdf = PdfPages("Multi_Output_Regression_Net.pdf")

# model instantiation and runs
multi_output = MultiOutput2LN(X, T, K)
loss_gd = multi_output.gradient_descent(eta=learning_rate, epochs=epochs)
loss_sgd = multi_output.stochastic_gradient_descent(eta=learning_rate, epochs=epochs, batchsize=batchsize)
loss_sgd_m = multi_output.stochastic_gradient_descent(eta=learning_rate, mu=0.5, epochs=epochs, batchsize=batchsize)

# plots
plt.plot(np.arange(loss_sgd_m.shape[0]), loss_sgd_m, '-', color="orange", label="Stochastic Gradient Descent + Momentum")
plt.plot(np.arange(loss_sgd.shape[0]), loss_sgd, 'c-', label=f"Stochastic Gradient Descent (B={batchsize})")
plt.plot(np.arange(loss_gd.shape[0]), loss_gd, 'g-', label="Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xscale("log")
plt.yscale("log")
plt.legend(loc="upper right", fontsize="medium")

pdf.savefig()
plt.close()
pdf.close()

# evaluation
names = ["sex", "paid classes", "romantic relationship", "daily alcohol"]
indices = [2, 18, 23, 27]
values = [[1,-1], [1,-1], [1, -1], range(1,6)]

for i in range(len(names)):
	multi_output.evaluate(X=X, index=indices[i], name=names[i], values=values[i])





