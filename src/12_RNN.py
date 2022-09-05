import torch
import numpy as np
import os
from os.path import exists
import requests
import collections
from sklearn.preprocessing import OneHotEncoder


# Global variables
# --------------------------------------
batch_size = 256
learning_rate = 0.001
seq_length = 20
epochs = 100
device = torch.device("cuda")


# Downloading the raw dataset
# --------------------------------------
path = "data/Shakespeare/"
file_name = "shakespeare"
file_ending = ".txt"

if not exists(path + file_name + file_ending):
	url = "https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt"
	response = requests.get(url, allow_redirects=True)

	with open(path + file_name + file_ending, 'w') as file_out:
		file_out.write(response.text)

	print(f"Content written into {path+file_name+file_ending}")


# Input processing
# --------------------------------------
# NOTE: Consider earlier commit for an alternative input processing (e.g. using re, replace(), etc.)
characters = set()

with open(path + file_name + file_ending, 'r') as file:
	for line in file:
		# note: rstrip() removes all trailing characters at the end of the line
		for el in line.lower().rstrip():
			# transform characters into its ordinal Unicode counterpart
			characters.add(ord(el))

# sort the set (-> returns a list)
characters = sorted(characters)
D = len(characters)

# association between chars and indices
char_dict = {char:idx for idx, char in enumerate(characters)}


# Classes and functions
# --------------------------------------
class RNN(torch.nn.Module):
	def __init__(self, D_):
		super(RNN, self).__init__()
		# weight matrices with 1000 features
		self.W1 = torch.nn.Linear(in_features=D_, out_features=1000)
		self.W2 = torch.nn.Linear(in_features=self.W1.out_features, out_features=D_)
		# recurrent module
		self.Wr = torch.nn.Linear(in_features=self.W1.out_features, out_features=self.W1.out_features)
		self.activation = torch.nn.PReLU()

	# recall: one sample consists of S=20 one-hot-vectors
	# input to the forward function is a batch consisting of b samples (-> 3D tensor)
	def forward(self, x):
		# initialise hidden vector per sample in the batch (h_s as 2D tensor)
		h_s = torch.zeros(len(x), self.W1.out_features, device=device)
		# structure for storing logits per sequence item
		Z = []
		for i in range(seq_length):
			# compute pre-activation and apply activation function
			# note: the linear layer self.W1 accepts multiple vectors as inputs
			a_s = self.W1(x[:,i]) + self.Wr(h_s)
			h_s = self.activation(a_s)

			# append logit value to dedicated data structure
			Z.append(self.W2(h_s))
		# note: transpose specifies which dimensions to tranpose first
		return torch.stack(Z).transpose(1, 0)
		
	def predict(self, x):
		# initialise hidden vector to zeros
		h_s = torch.zeros(self.W1.out_features, device=device)
		# note: the size of x might not be the same as in training
		for i in range(x.shape[1]):
			# compute pre-activation and apply activation function
			a_s = self.W1(x[:,i]) + self.Wr(h_s)
			h_s = self.activation(a_s)
		# return only logit of the last character
		return self.W2(h_s)


def one_hot_batch(batch_):
	# note: a batch consists of a set of sequences with every sequence being a list of indices (from the lookup dict)
	# a batch comes in form of a 2D torch tensor (see function train_network() below!)
	batch_oneHot = torch.zeros((batch_.shape[0], batch_.shape[1], D), device=device)
	for i in range(batch_.shape[0]):
		for j in range(batch_.shape[1]):
			# ignore unknown values (zero-padding)
			if batch_[i, j] >= 0:
				batch_oneHot[i, j, batch_[i, j]] = 1
	return batch_oneHot


def train_network():
	network = RNN(D).to(device)
	# cross-entropy loss: ignore_index specifies a target value that is ignored
	loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
	# SGD
	optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

	# create dataset tensors by using deques (-> doubly ended queue as more flexible version of a list)
	# Note: if deque with given maxlen is full, leading entries are simply popped
	data = collections.deque(maxlen=seq_length)
	# instantiate deque with -1 (zero-padding -> does not do anything

	data.extend([-1] * seq_length)
	# X and T as lists of np arrays
	X, T = [], []

	with open(path + file_name + file_ending, 'r') as file:
		for line in file:
			# ignoring empty lines
			if not line.rstrip():
				continue
			# iterating through all the elements in the line
			for el in line.replace("\n", " ").lower():
				X.append(np.array(data))
				# append the index that represents the current (ordinally encoded) char to the dequeue
				data.append(char_dict[ord(el)])
				# current target
				T.append(np.array(data))
	print(f"Created dataset of {len(T)} samples with input size {seq_length}x{D}")

	# dataset and data loader from tensor (use shuffling)
	dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(T, dtype=torch.long))
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	# training
	try:
		for epoch in range(epochs):
			# measuring training loss (no validation set)
			loss_total = 0.
			# iterating over the batches
			for x,t in data_loader:
				t = t.to(device)
				optimizer.zero_grad()
				# forward pass
				z = network(one_hot_batch(x))
				# computing average loss
				J = torch.stack([loss(z[:,i], t[:,i]) for i in range(seq_length)]).sum()
				J.backward()
				optimizer.step()
				# add up total loss
				loss_total += J.cpu().detach().item()
				print(f"\rLoss: {float(J)/t.shape[0]: 3.5f}", end="")
			print(f"\rEpoch: {epoch} -- Loss: {loss_total/len(dataset)}")

			# save model after each epoch
			torch.save(network.state_dict(), path+"text.model")

	except KeyboardInterrupt:
		print()

	return network

def load_model():
	network = RNN(D)
	network.load_state_dict(torch.load(path+"text.model"))
	return network.to(device)


# main functionality
# --------------------------------------
if __name__ == "__main__":
	import sys

	# first option: "train", "best" (get char with highest probability), others: sample char based on probabilies
	option = sys.argv[1] if len(sys.argv) > 1 else "best"
	samples = sys.argv[2] if len(sys.argv) > 2 else ("the ", "beau", "mothe", "bloo")

	if option == "train":
		network = train_network()

	else:
		network = load_model()

		# go over all seeds
		for seed in samples:
			text = seed

			with torch.no_grad():
				# adding 80 characters
				for i in range(80):
					# turn current text to one-hot batch
					x = one_hot_batch(np.array([[char_dict[ord(i)] for i in text]]))
					# predict the next char
					z = network.predict(x)
					y = torch.softmax(z,1).cpu().numpy()

					# select according to sampling scheme
					if option == "best":
						#take char associated with highest probability
						next_char = characters[np.argmax(y)]
					else:
						# sample chars based on probability distribution
						next_char = random.choices(characters, y[0])[0]
					# append character to text
					text  = text + chr(next_char)

				# print seed and text
				print(f"{seed} -> \"{text}\"")












