from pathlib import Path
import sys

root_directory = Path(__file__).parent.parent.resolve()
print(root_directory)
# adding stargan folder with source code to system path (-> access to model)
sys.path.insert(0, root_directory / "src" / "stargan")

from skimage import io 
import torch
from torch.utils import data
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image

from stargan.model import Generator


####################################################################################################
# README:

# This exercise is based on the paper by Choi et al., 2018 and their proposed StarGAN network 
# for image domain transfers. The current code uses the StarGAN code provided by the authors on
# their github repo: https://github.com/yunjey/stargan

# To run this script, please specify/customise the following variables:

img_names_in = ['portrait_1.jpg', 'portrait_2.jpg'] # filenames of the to-be-manipulated portrait images (must be stored in data/stargan/). Note that the portrait images must be of the same person in order to properly work.
crop_loc = [[0,1700,203,1597], [0,1000,70,890]] # locations where input portraits must be cropped
attributes = [0., 0., 1., 1., 1.] # vector that indicates if the facial attributes [black hair, blond hair, brown hair, male, young] of the portrayed person are true (=1) or not (=0).

# Note: The generated images can be found in the directory results/stargan.
####################################################################################################


# class and function definitions
# ------------------------------------------------------------
class PortraitTest(data.Dataset):
	# dataset class for the portraits to be translated
	# Note: Minimal implementation only for test mode on fixed selected attributes

	def __init__(self, img_dir, img_names, attr_vec, transform):
		assert (len(attr_vec) == 5), "Invalid number of elements in attribute vector (-> required: 5)"
		self.img_dir = img_dir
		self.img_names = img_names
		self.attr_vec = attr_vec
		self.selected_attr = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
		self.transform = transform
		self.dataset = []
		self.prepareItems()

	def prepareItems(self):
		for i, name in enumerate(self.img_names):
			target_domains = []
			for idx, attr in enumerate(self.selected_attr):
				attr_vec = self.attr_vec.copy()
				if attr in self.selected_attr[0:3]:
					for j in range(3):
						attr_vec[j] = 0 if j != idx else 1
				else:
					# flip entry
					attr_vec[idx] = 0 if attr_vec[idx] == 1 else 1
				target_domains.append(torch.FloatTensor(attr_vec))
			self.dataset.append([name, target_domains])

	def __getitem__(self, idx):
		filename, target_domains = self.dataset[idx]
		image = Image.open(self.img_dir / filename)
		return self.transform(image), target_domains

	def __len__(self):
		return len(self.dataset)


def build_model():
    # creating the generator (minimal implementation acc. to task description)
    gen = Generator()
    return gen


def restore_model(generator, model_save_dir, resume_iters=200000):
    # restoring the generator (minimal implementation acc. to task description)
    print('Loading the trained models from step {}...'.format(resume_iters))
    G_path = model_save_dir / '{}-G.ckpt'.format(resume_iters)
    generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))


def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)


# main
# ------------------------------------------------------------
img_names_out = []
Path(root_directory / "data" / "stargan").mkdir(parents=True, exist_ok=True)
Path(root_directory / "results" / "stargan").mkdir(parents=True, exist_ok=True)
path_in = root_directory / "data" / "stargan"
path_out = root_directory / "data" / "stargan"

for idx, img_name in enumerate(img_names_in):
	img_path = path_in / img_name
	img = io.imread(img_path)
	crop = crop_loc[idx]
	img = img[crop[0]:crop[1], crop[2]:crop[3]]
	img_names_out.append(str(idx+1).zfill(6) + '.jpg')

	io.imsave(path_out / img_names_out[idx], img)

model_save_dir = root_directory / "src" / "stargan" / "stargan_celeba_128" / "models"
result_dir = root_directory / "results" / "stargan"
image_size = 128

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# transform used by StarGAN source code
transform = []
transform.append(T.Resize(image_size))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

dataset = PortraitTest(path_out, img_names_out, attributes, transform)
data_loader = data.DataLoader(dataset=dataset)

# instantiating generator
generator = build_model()
generator.to(device)

# restoring pre-trained generator
restore_model(generator, model_save_dir)

with torch.no_grad():
	for idx, (img, target_domains) in enumerate(data_loader):
		# send input images to device
		img = img.to(device)

		# translate images
		img_fake_list = [img]
		for target in target_domains:
			img_fake_list.append(generator(img, target.to(device)))

		# save translated images
		img_concat = torch.cat(img_fake_list, dim=3)
		result_path = result_dir / '{}-images.jpg'.format(idx+1)
		save_image(denorm(img_concat.data.cpu()), result_path, nrow=1, padding=0)
		print('Saved real and fake images into {}...'.format(result_path))