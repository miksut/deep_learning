import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, ToPILImage
import torchvision.models as models
import matplotlib as plt
import numpy as np
import requests
import zipfile
import os
from os.path import exists
from PIL import Image
import types
from scipy.spatial.distance import euclidean


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

	def __getitem__(self, idx=None, img_name=None):
		image = None

		if idx:
			image_name = self.imageNames[idx]
			image = Image.open(self.root + "/" + image_name)

		if img_name:
			image = Image.open(self.root + "/" + img_name) 

		if self.transform:
			image = self.transform(image)
			image = image.repeat(3,1,1)
			image = image.unsqueeze(0)

		return image


# Redefining forward implementation
# --------------------------------------
def customized_forward(self, x):
	x = self.conv1(x)
	x = self.bn1(x)
	x = self.relu(x)
	
	x = self.maxpool(x)
	x = self.layer1(x)
	x = self.layer2(x)
	x = self.layer3(x)
	x = self.layer4(x)

	x = self.avgpool(x)
	x = torch.flatten(x, 1)

	# convert tensor into numpy array
	x = x.detach().numpy()
	x = np.squeeze(x, 0)

	return x


# main
# --------------------------------------
# global variables
keys = ["normal", "happy", "sad", "sleepy", "surprised", "wink", "glasses", "noglasses", "leftlight", "rightlight", "centerlight"]
subjects = 15
feature_dim = [subjects, 512]
values = [np.zeros([feature_dim[0], feature_dim[1]]) for i in range(len(keys))]
res = [0 for i in range(len(keys))]

# Check for GPU availability (-> training procedure)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# initialise transformations and dataset
transformations = Compose([Resize(300), CenterCrop(224), ToTensor()])
data = FaceDataset(root=path, transform=transformations)

# load pre-trained model
model = models.resnet18(pretrained=True)

# bind customised forward implementation to object
model._forward_impl = types.MethodType(customized_forward, model)

# set modules to evaluation mode
model.eval()

# initialise data structure that stores the deep features via dict comprehension
features = {k:v for (k,v) in zip(keys, values)}

# initialise data structure for results
results = {k:v for (k,v) in zip(keys, res)}

# populate the data structure with features
for i in keys:
	for j in range(subjects):
		img = None
		if j<=8:
			img = data.__getitem__(img_name="subject" + "0"+ str(j+1) + "." + i)
		else:
			img = data.__getitem__(img_name="subject" + str(j+1) + "." + i)

		features[i][j] = model(img)		

# evaluation
for i in keys: 
	for j in range(subjects):
		min_distance = np.inf
		idx = None
		for k in range(subjects):
			dist = euclidean(features["normal"][k], features[i][j])
			if dist <= min_distance:
				min_distance = dist
				idx = k
		if j == idx:
			results[i] += 1

print(results)
