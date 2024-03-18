import os
import zipfile
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from torchvision import models
from torchvision.utils import make_grid
from torchvision import transforms as tsfm
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image
from pathlib import Path
from IPython import display

data_dir = '/data/Han212/ML/CNN_Plant_Seedling_Classification/datasets/' 

#if not os.path.exists(data_dir):
#  zip_dir = '/data/Han212/ML/CNN_Plant_Seedling_Classification/datasets/plant-seedlings-classification.zip'
#
#  with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
#      zip_ref.extractall(data_dir)
      
      
class Train_data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root=root_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

class Pred_data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = list(Path(root_dir).glob('*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img
        
## visualize dataset item for debug
#train_dir = os.path.join(data_dir, 'train')
#test_dir = os.path.join(data_dir, 'test')
#transform = tsfm.Compose([
#    tsfm.Resize((224, 224)),
#    tsfm.ToTensor(),
#])
#
#whole_set = Train_data(
#    root_dir=train_dir,
#    transform=transform
#)
#
#test_set = Pred_data(
#    root_dir=test_dir,
#    transform=transform
#)
#
#num_images_to_display = 5
#fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 3))
#
#for i, (img, label) in enumerate(whole_set):
#    axs[i].imshow(img.permute(1, 2, 0))
#    axs[i].set_title(f'Class: {label}')
#    axs[i].axis('off')
#
#    num_images_to_display -= 1
#    if num_images_to_display == 0:
#        break
#
#plt.tight_layout()
#plt.show()
#
#num_images_to_display = 5
#fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 3))
#for i, img in enumerate(test_set):
#    axs[i].imshow(img[0].permute(1, 2, 0))
#    axs[i].set_title(f'Test img: {i}')
#    axs[i].axis('off')
#
#    num_images_to_display -= 1
#    if num_images_to_display == 0:
#        break
#
#plt.tight_layout()
#plt.show()