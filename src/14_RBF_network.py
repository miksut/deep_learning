from pathlib import Path
import torch
from torchvision.datasets import MNIST
import torchvision
import random
import numpy as np

torch.manual_seed(0)
root_directory = Path(__file__).parent.parent.resolve()
(root_directory / "results" / "rbf_network").mkdir(parents=True, exist_ok=True)


# global variables
# --------------------------------------
device = torch.device("cuda")
learning_rate = 1e-4
epochs = 100
df_dim = 2 # deep feature dimensionality
no_prototypes = 100 # number of prototypes for RBF layer


# RBF layer
# --------------------------------------
class RBF(torch.nn.Module):
	def __init__(self, no_prototypes, sample_dim):
		super(RBF, self).__init__()

		self.D = sample_dim
		self.K = no_prototypes
		self.W = torch.nn.Parameter(torch.zeros(no_prototypes, sample_dim))
		torch.nn.init.normal_(self.W, 0, 1)
		self.variances = torch.nn.Parameter(torch.ones(no_prototypes))

	def forward(self, x):
		B = x.shape[0]
		W = self.W.unsqueeze(0).expand(B, self.K, self.D)
		X = x.unsqueeze(1).expand(B, self.K, self.D)
		# note: -1 refers to the last dimension in the 3d tensor
		A = torch.sum(torch.pow(W-X, 2), -1)

		result = torch.exp(-A/self.variances)
		return result


# RBF layer
# --------------------------------------
class RBFNetwork(torch.nn.Module):
	def __init__(self, no_hidden, no_prototypes):
		super(RBFNetwork, self).__init__()

		# define network architecture
		self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1, padding=2)
		self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
		self.pool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
		self.activation = torch.nn.Sigmoid()
		self.bn = torch.nn.BatchNorm2d(self.conv2.out_channels)
		self.fc1 = torch.nn.Linear(7*7*64, no_hidden, bias=True)
		self.rbf = RBF(no_prototypes, no_hidden)
		self.fc2 = torch.nn.Linear(no_prototypes, 10, bias=True)

	def forward(self, x):
		a = self.extract(x)
		return self.fc2(self.rbf(a))		

	# note: this allows us to extract deep features on demand
	def extract(self, x):
		a = self.activation(self.pool(self.conv1(x)))
		a = self.activation(self.bn(self.pool(self.conv2(a))))
		a = torch.flatten(a, 1)
		return self.fc1(a)


# preparing datasets
# --------------------------------------
train_set = MNIST(root=root_directory / "data", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=32)

test_set = MNIST(root=root_directory / "data", train=False, download=True, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=32)


# training
# --------------------------------------
network = RBFNetwork(df_dim, no_prototypes)
network = network.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=network.parameters(), lr=learning_rate, momentum=0.9)

# training network
best = 0
torch.save(network.state_dict(), root_directory / "results" / "rbf_network" / "Init.model")

print(f'Start Training:')
print(f'Learning Rate: {learning_rate} \t Epochs: {epochs} \t Number of Prototypes: {no_prototypes}\n')
for epoch in range(epochs):
	for x,t in train_loader:
		optimizer.zero_grad()
		Z = network(x.to(device))
		J = loss(Z, t.to(device))
		J.backward()
		optimizer.step()

	# computing test accuracy
	correct = 0
	with torch.no_grad():
		for x,t in test_loader:
			z_test = network(x.to(device))
			# compute indices of largest logits per sample
			y_pred = torch.argmax(z_test, dim=1)
			correct += (y_pred == t.to(device)).type(torch.float).sum().item()

	# print epoch and accuracy
	print(f"Epoch: {epoch+1}; test accuracy: {correct/len(test_set)*100.:1.2f} %")
	if correct > best:
		best = correct
		torch.save(network.state_dict(), root_directory / "results" / "rbf_network" / "Best.model")
