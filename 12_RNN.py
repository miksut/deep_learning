import torch
from torch.utils import data
import numpy as np
import os
from os.path import exists
import requests
import re
from sklearn.preprocessing import OneHotEncoder


# Classes and functions
# --------------------------------------
class SequenceDataset(data.Dataset):
	def __init__(self, X_, T_):
		assert(X_.shape == T_.shape), "Non-matching dimensions of data and targets"
		self.X = torch.DoubleTensor(X_)
		self.T = torch.DoubleTensor(T_)

	def __getitem__(self, idx):
		return self.X[idx], self.T[idx]

	def __len__(self):
		return self.X.shape[0]


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
data_raw =""
replacements = 0

with open(path + file_name + file_ending, 'r') as file:
	for line in file:
		current_line = line.replace('\n', ' ').lower()
		data_raw += current_line

# remove redundant spaces
data_raw = re.sub(' +', ' ', data_raw)

# OneHot encoding of all chars occuring in text
char_set = set(data_raw)
char_arr = np.array(list(char_set))

enc = OneHotEncoder(sparse=False)
oneHot_enc = enc.fit_transform(char_arr.reshape(-1,1))

# dictionaries for lookups: char->OneHot and OneHot->char
char_dict = {el : oneHot_enc[np.where(char_arr == el)[0][0]] for el in char_arr}
oneHot_dict = {val.tobytes() : key for key, val in char_dict.items()}

# OneHot-encode text string
data_enc = np.zeros([len(data_raw), len(char_arr)])

for i in range(len(data_raw)):
	data_enc[i] = char_dict[data_raw[i]]


# sequence encoding: preparing samples
# --------------------------------------
seq_length = 20
n_samples = len(data_raw)-seq_length
X = np.zeros([n_samples,seq_length,len(char_arr)])
T = np.zeros([n_samples,seq_length,len(char_arr)])

for i in range(n_samples):
	X[i] = data_enc[i : i+seq_length]
	T[i] = data_enc[i+1 : i+1+seq_length]

# instantiating datset and dataloader
dataset = SequenceDataset(X, T)
data_loader = data.DataLoader(dataset=dataset)

#print(oneHot_dict[X_torch[-1][-1].numpy().tobytes()])


