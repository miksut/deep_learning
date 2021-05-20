import torch
import numpy as np
import os
from os.path import exists
import requests
import re
from sklearn.preprocessing import OneHotEncoder





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

