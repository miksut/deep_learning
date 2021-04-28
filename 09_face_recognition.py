import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, ToPILImage
import matplotlib as plt
import numpy as np
import requests
import zipfile
import os
from os.path import exists
from PIL import Image

# Preparing the raw dataset
# --------------------------------------
path = "data/Faces/yalefaces"
file_ending = ".zip"

if not exists(path + file_ending):
	url = "http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip"
	req = requests.get(url, allow_redirects=True)
	open(path + file_ending, "wb").write(req.content)

	zip_file = path + file_ending
	try:
		with zipfile.ZipFile(zip_file, "r") as z:
			z.extractall(path + "/..")
			print("Unzipping successful")

	except:
		print("Invalid file")

# Removing unnecessary files from the raw dataset and enforcing consistent naming
files_del = ["Readme.txt", "Subject01.glasses.gif"]
for file in files_del:
	if exists(path + "/" + file):
		os.remove(path + "/" + file)

if exists(path + "/" + "subject01.gif"):
	os.rename(path + "/" + "subject01.gif", path + "/" + "subject01.centerlight")

# Dedicated class for the dataset
# --------------------------------------
class FaceDataset(Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.transform = transform
		self.imageNames = next(os.walk(root))[2]

	def __len__(self):
		return len(next(os.walk(self.root))[2])

	def __getitem__(self, idx):
		image_name = self.imageNames[idx]
		image = Image.open(self.root + "/" + image_name)

		if self.transform:
			image = self.transform(image)


		return image







# main
# --------------------------------------
transformations = Compose([Resize(300), CenterCrop(224), ToTensor()])
data = FaceDataset(root=path, transform=transformations)
element = data[3]

print(element.shape)

face_dataloader = DataLoader(data, batch_size=15, shuffle=False)

