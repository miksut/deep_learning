import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# preparing training dataset
training_data = MNIST(root='data', download=True, train=True, transform=Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))]))
test_data = MNIST(root='data', download=True, train=False, transform=Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))]))

print(training_data[0][0].shape)
print(test_data[0][0].shape)

# define model
class CatClassifier(nn.Module):
	def __init__(self, K):
		super(CatClassifier, self).__init__()
		self.cat_clf_stack = nn.Sequential(
			nn.Linear(training_data[0][0].shape[0], K, bias=False),
			nn.Sigmoid(),
			nn.Linear(K, 10)
		)

	def forward(self, x):
		logits = self.cat_clf_stack(x)
		return logits

# function for training procedure
def train(data_loader, model, loss_function, optimizer):
	size = len(data_loader.dataset)

	for batch, (X,y) in enumerate(data_loader):
		X,y = X.to(device), y.to(device)

		# Compute loss
		pred = model(X)
		loss = loss_function(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f} [{current:>5d}/{size*len(X):>5d}]")

def test(data_loader, model, loss_function):
	# set modules in evaluation mode
	model.eval()
	test_loss, correct = 0,0
	# deactivating autograd engine (-> memory)
	with torch.no_grad():
		for X, y in data_loader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_function(pred, y).type(torch.float).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= len(data_loader)
	correct /= len(data_loader.dataset)
	print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg. Loss: {test_loss:>0.8f} \n")
	return test_loss, 100*correct

# main 
# ------------------------------------------------------
# global variables
hidden_neurons = 25
batch_size = 64
epochs = 100
accuracy_max = [0.0, 0.0]
accuracies = np.arange(epochs, dtype=float)
losses = np.arange(epochs, dtype=float)

# Check for GPU availability (-> training procedure)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X,y in train_dataloader:
	print(f"Shape of X: {X.shape}")
	print(f"Shape of y: {y.shape}")
	break

# instantiating the model
model = CatClassifier(hidden_neurons).to(device)

# define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# train and evaluate model
for epoch in range(epochs):
	print(f"Epoch {epoch+1}\n-------------------------------")
	train(train_dataloader, model, loss_fn, optimizer)
	loss, accuracy = test(test_dataloader, model, loss_fn)
	losses[epoch] = loss
	accuracies[epoch] = accuracy

	if accuracy > accuracy_max[0]:
		accuracy_max[0] = accuracy
		accuracy_max[1] = epoch

print(f"Highest accuracy in epoch {accuracy_max[1]+1}: {accuracy_max[0]}% \n")


# Plotting
root_directory = Path(__file__).parent.parent.resolve()
pdf = PdfPages(root_directory / "results" / "07_PyTorch_CatClf.pdf")
fig, ax = plt.subplots()
ax.plot(np.arange(epochs)+1., losses, '-', color='blue')
ax.set_xscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Avg. Loss")
ax.yaxis.label.set_color("blue")
ax.tick_params(axis='y', colors="blue")
ax.set_title("Loss and Accuracy Progression on MNIST")

ax2 = ax.twinx()
ax2.plot(np.arange(epochs)+1., accuracies, '-', color='red')
ax2.scatter(accuracy_max[1]+1, accuracy_max[0], color='red')
ax2.set_ylabel("Accuracy")
ax2.yaxis.label.set_color("red")
ax2.tick_params(axis='y', colors="red")

pdf.savefig()
plt.show()
pdf.close()
