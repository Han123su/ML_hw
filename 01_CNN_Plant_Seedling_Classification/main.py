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

from dataset import *
from models import *
from train import *
from valid import *
from plot_and_predict import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '/data/Han212/ML/CNN_Plant_Seedling_Classification/datasets/' 

# Set Hyperparameters
batch_size = 64
epochs = 40
learning_rate = 0.001
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# initial transform
transform = tsfm.Compose([
    tsfm.Resize((224, 224)),
    tsfm.ToTensor(),
])

# initial dataset
whole_set = Train_data(
    root_dir=train_dir,
    transform=transform
)

test_set = Pred_data(
    root_dir=test_dir,
    transform=transform
)

# split train valid and initial dataloader
train_set_size = int(len(whole_set) * 0.8)
valid_set_size = len(whole_set) - train_set_size
train_set, valid_set = random_split(whole_set, [train_set_size, valid_set_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# initial model
model = resnet_50(num_classes=12).cuda()   # model = resnet_50(num_classes=12).to(device)

# initial loss_function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# initial plot values
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []
epoch_list = []

# repeat train and valid epochs times
print(epochs)
for epoch in range(epochs):
  epoch_list.append(epoch + 1)

  loss, acc = train(
      model,
      criterion,
      optimizer,
      train_loader,
      epoch=epoch,
      total_epochs=epochs,
      batch_size=batch_size
  )
  train_loss.append(loss)
  train_acc.append(acc)
  print(f'Avg train Loss: {loss}, Avg train acc: {acc}')

  loss, acc = valid(
      model,
      criterion,
      valid_loader,
      epoch=epoch,
      total_epochs=epochs,
      batch_size=batch_size
  )
  valid_loss.append(loss)
  valid_acc.append(acc)
  print(f'Avg valid Loss: {loss}, Avg valid acc: {acc}')

Plot("Loss Curve", 'Loss', epoch_list, train_loss, valid_loss)
Plot("Accuarcy Curve", 'Acc', epoch_list, train_acc, valid_acc)

preds = predict(test_set, model)
view_pred_result(test_set, preds)
