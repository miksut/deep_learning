from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np

# global variables
# --------------------------------------
threshold = 0.5
torch.manual_seed(0)
device = torch.device('cuda')
threshold = 0.5


# dedicated class for one-hot vector 
# --------------------------------------
class TargetVector():
	# idea: define known, known unknown and unknown classes. The first two are used for training
	def __init__(self, known_targets=(4,5,8,9), unknown_targets=(0,2,3,7), ignored_targets=(1,6)):
		self.known_targets = known_targets
		self.unknown_targets = unknown_targets
		self.ignored_targets = ignored_targets

		# prepare data structure for accessing one-hot vectors for known classes
		self.one_hot_known = np.eye(len(self.known_targets))
		self.t_known = {c : self.one_hot_known[idx] for idx, c in enumerate(self.known_targets)}

		# prepare one-hot vector for known unknown classes
		self.t_unknown = np.ones(len(known_targets)) / len(known_targets)

	# given an input batch, return filtered batch (known and known unknonw) with one-hot target vector
	def __call__(self, inputs, targets):
		# split off the unknown classes
		valid_inputs = []
		valid_targets = []

		for idx, t in enumerate(targets):
			# check for known classes
			if t in self.known_targets:
				valid_inputs.append(inputs[idx].numpy())
				valid_targets.append(self.t_known[int(t)])
			# check for known unkown classes
			elif t in self.unknown_targets:
				valid_inputs.append(inputs[idx].numpy())
				valid_targets.append(self.t_unknown)

		# return filtered batch with valid data as two tensors
		return torch.tensor(valid_inputs), torch.tensor(valid_targets)


	# computes the predicted class and the associated confidence
	def predict(self, logits):
		# softmax over logits
		SoftMax = torch.nn.Softmax(dim=1)
		confidences = SoftMax(logits)
		# indices of the class with maximal confidence (= predictions)
		indices = torch.argmax(logits, dim=1)
		# filter the max confidence vales per batch-sample
		max_confidence = confidences[range(len(logits)), indices]

		# return a tuple containing predicted class and associated confidence value per batch-sample
		return [(self.known_targets[indices[i]], max_confidence[i]) for i in range(len(logits))]


	# computes the confidence metric per batch (used in evaluation later) 
	def confidence(self, logits, targets):
		SoftMax = torch.nn.Softmax(dim=1)
		confidences = SoftMax(logits).numpy()

		# return confidence of correct class for known samples and 1.0 - max(confidences) + 1.0/O for unknown samples
		return [
			# known targets
			np.sum(confidences[i] * self.t_known[int(targets[i])])
				if targets[i] in self.known_targets
				# unknown targets
				else 1.0 - np.max(confidences[i]) + (1.0/len(self.known_targets))
			# iterate over batch using list comprehension
			for i in range(len(logits))
		]


# define autograd function for custom loss implementation
# --------------------------------------
class AdaptedSoftMax(torch.autograd.Function):

	# implementing the forward pass
	@staticmethod
	def forward(ctx, logits, targets):
		# compute log probabilities via log softmax
		logSoftMax = torch.nn.LogSoftmax(dim=1)
		y_log = logSoftMax(logits)
		# save log probabilities and targets for backward computation
		ctx.save_for_backward(y_log, targets)
		# compute loss
		loss = - torch.mean(y_log * targets)
		return loss

	@ staticmethod
	def backward(ctx, result):
		# get results stored from forward pass
		y_log, targets = ctx.saved_tensors
		# compute probabilities from log probabilities
		y = torch.exp(y_log)
		return y - targets, None

# Network implementation
# --------------------------------------
class Convolutional(torch.nn.Module):
	def __init__(self, K, O):
		super(Convolutional, self).__init__()
		# define convolutional layers
		self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1, padding=2)
		self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=32, kernel_size=(5,5), stride=1, padding=2)
		self.pool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
		self.activation = torch.nn.Sigmoid()
		self.bn = torch.nn.BatchNorm2d(self.conv2.out_channels)
		self.fc1 = torch.nn.Linear(7*7*32, K, bias=True)
		self.fc2 = torch.nn.Linear(K, O)

	def forward(self, x):
		a = self.activation(self.pool(self.conv1(x)))
		a = self.activation(self.bn(self.pool(self.conv2(a))))
		a = torch.flatten(a,1)
		return self.fc2(self.activation(self.fc1(a)))

# training and test set
root_directory = Path(__file__).parent.parent.resolve()
train_set = MNIST(root=root_directory / "data" / "MNIST" / "raw", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_set, shuffle=True, batch_size=32)

test_set = MNIST(root=root_directory / "data" / "MNIST" / "raw", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_set, shuffle=True, batch_size=32)

#instantiate network, loss, and optimizer
network = Convolutional(50, 4).to(device)
loss = AdaptedSoftMax.apply
optimizer = torch.optim.SGD(params=network.parameters(), lr=0.1, momentum=0.9)

# instance of class for one-hot encoding
targets = TargetVector()

# train for several epochs
for epoch in range(100):
	for x,t in train_loader:
		# convert targets and filter unknown unknowns
		x,t = targets(x,t)

		z = network(x.to(device))
		J = loss(z, t.to(device))
		J.backward()
		optimizer.step()

	# evaluation after each epoch
	k, ku, uu = 0, 0, 0
	nk, nku, nuu = 0, 0, 0
	# average confidence
	conf = 0.

	with torch.no_grad():
		for x,t in test_loader:
			# compute network output
			z = network(x.to(device)).cpu()
			# compute predicted classes and their confidences
			predictions = targets.predict(z)
			# add confidence metric for batch
			conf += np.sum(targets.confidence(z,t))
			# compute accuracy
			for i in range(len(t)):
				if t[i] in targets.known_targets:
					if predictions[i][0] == int(t[i]) and predictions[i][1] >= threshold:
						k += 1
					nk += 1
				elif t[i] in targets.unknown_targets:
					if predictions[i][1] < threshold:
						ku += 1
					nku += 1
				else:
					if predictions[i][1] < threshold:
						uu += 1
					nuu += 1

	# print epoch and metrics 
	print(f"Epoch {epoch+1}; test known: {k/nk*100.:1.2f} %, known unknown: {ku/nku*100.:1.2f} %, unknown unknown: {uu/nuu*100.:1.2f} %; average confidence: {conf/len(test_set):1.5f}")
	print()
