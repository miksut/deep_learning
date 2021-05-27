import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# global variables
# --------------------------------------
batch_size = 256
device = "cuda"
epochs = 100
learning_rate = 0.001
path = "data/MNIST/"
method = "FGS"
# set seed for reproducibility
torch.manual_seed(0)


# command line parser object
# --------------------------------------
def command_line_options():
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# adding arguments
	parser.add_argument("-t", "--train", default=False, help="Train the model (without adversarial training)")
	parser.add_argument("-a", "--train_adversarial", default=False, help="Adversarial model training")
	parser.add_argument("-g", "--generate", default=False, help="Generating and evaluating adversarial samples")

	return parser.parse_args()


# loading data and preparing data loader
# --------------------------------------
training_data = MNIST(root='data', train=True, download=False, transform=ToTensor())
test_data = MNIST(root='data', train=False, download=False, transform=ToTensor())

loader_train = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
loader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


# model
# --------------------------------------
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.CNN_stack = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),

			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),

			nn.Flatten(),
			nn.Linear(32*7*7, 10)
		)
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

	def forward(self, X):
		logits = self.CNN_stack(X)
		return logits

	def FGS(self, X, T, alpha=0.3):
		# enable gradient for input
		X.requires_grad_(True)

		# compute loss and gradient
		Z = self.forward(X)
		J = self.loss(Z, T)
		J.backward()

		X_adv = torch.clamp(X + alpha*(torch.sign(X.grad)), min=0., max=1.)
		return X_adv

	def FGV(self, X, T, alpha=0.6):
		# enable gradient for input
		X.requires_grad_(True)

		# compute loss and gradient
		Z = self.forward(X)
		J = self.loss(Z, T)
		J.backward()

		max_abs = torch.max(torch.abs(X.grad))
		X_adv = torch.clamp(X + alpha*(X.grad/max_abs), min=0., max=1.)
		return X_adv


# testing/evaluation function
# --------------------------------------
def eval_network(network, adv=False, method="FGS"):
	# set modules in evaluation mode
	network.eval()

	test_loss, correct_clf = 0., 0.

	if adv==True:
		correct_clf_adv = 0.
		samples_adv = 0.

	for x,t in loader_test:
		torch.set_grad_enabled(False)
		x,t = x.to(device), t.to(device)

		Z = network(x)
		J_test = network.loss(Z, t)

		test_loss += J_test.type(torch.float).item()
		correct_clf += (Z.argmax(dim=1) == t).type(torch.float).sum().item()

		if adv==True:
			network.train()
			torch.set_grad_enabled(True)
			X_adv = network.FGS(x, t) if method=="FGS" else network.FGV(x, t)
			network.eval()

			# only use adversarial samples for correctly classified original samples
			relevant_samples = (Z.argmax(dim=1) == t)
			X_adv = X_adv[relevant_samples]

			# compute logits and classify adversarial samples
			Z_adv = network(X_adv)
			correct_clf_adv += (Z_adv.argmax(dim=1) == t[relevant_samples]).type(torch.float).sum().item()
			samples_adv += X_adv.shape[0]
	
	torch.set_grad_enabled(True)
	if adv==False:
		return (test_loss/len(loader_test)), (correct_clf/len(loader_test.dataset))
	else:
		return (correct_clf/len(loader_test.dataset)), (correct_clf_adv/samples_adv)


# training function
# --------------------------------------
def train_network():
	accuracy_top = 0.
	network = CNN().to(device)

	# training
	try:
		for epoch in range(epochs):
			loss_total = 0.

			for x,t in loader_train:
				x,t = x.to(device), t.to(device)

				network.optimizer.zero_grad()
				Z = network(x)
				J = network.loss(Z, t)

				# returns the averaged loss per batch
				J.backward()
				network.optimizer.step()

				loss_total += J.cpu().detach().item()
				print(f"\rBatch Loss: {float(J):3.5f}", end="")
			print(f"\rEpoch: {epoch} --- Training Loss: {loss_total/len(loader_train)}")

			# evaluate model on test dataset
			test_loss, accuracy = eval_network(network)
			print(f"\rEpoch: {epoch} --- Test Loss: {test_loss:3.5f} --- Test Accuracy: {accuracy}")

			if accuracy > accuracy_top:
				accuracy_top = accuracy
				# save best model based on evaluations on test set
				torch.save(network.state_dict(), path+"CNN_MNIST.model")

	except KeyboardInterrupt:
		print()

	print(f"Saved model with top accuracy: {accuracy_top}")

	return network


# load existing model
# --------------------------------------	
def load_model():
	network = CNN()
	network.load_state_dict(torch.load(path+"CNN_MNIST.model"))
	print(f"Loaded model: {path}CNN_MNIST.model")
	return network.to(device)


# main functionality
# --------------------------------------

if __name__ == "__main__":
	import sys

	args = command_line_options()

	if args.train == "True":
		print(f"Train CNN w/o adversarial examples")
		print(f"Epochs: {epochs} --- Batch size: {batch_size} --- Learning rate: {learning_rate}")

		network = train_network()

	if args.generate == "True":
		network = load_model()
		acc, acc_adv = eval_network(network, adv=True, method=method)

		print(f"Accuracy on test set using original samples: {acc*100} %")
		print(f"Accuracy on test set using adversarial samples ({method}): {acc_adv*100} %")


