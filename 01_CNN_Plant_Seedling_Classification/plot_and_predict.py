import os
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


def Plot(title, ylabel, epochs, train_loss, valid_loss):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(ylabel)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.legend(['train', 'valid'], loc='upper left')
    

## debug "Plot" function
#debug_epochs = [1, 2, 3, 4, 5]
#debug_train_loss = [0.1, 0.08, 0.06, 0.05, 0.04]
#debug_valid_loss = [0.2, 0.15, 0.12, 0.1, 0.09]
#
#Plot("Training and Validation Loss", 'Loss', debug_epochs, debug_train_loss, debug_valid_loss)
#
#plt.show()

#################################################################################################

def predict(loader, model):
    model.eval()
    preds = []
    for data in tqdm(loader):
        pred = model(data.cuda())
        cls = torch.argmax(pred, dim=1)
        preds.append(cls)

    return preds
    
# Visualize Predict result
def view_pred_result(test_set, preds, num_images_to_display=5):
    labels = ['Black-grass', 'Charlock' , 'Cleavers' , 'Common Chickweed' , 'Common wheat' , 'Fat Hen' , 'Loose Silky-bent' , 'Maize' , 'Scentless Mayweed' , 'Shepherds Purse', 'Small-flowered Cranesbill' , 'Sugar beet']
    fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 3))
    for i, img in enumerate(test_set):
        axs[i].imshow(img[0].permute(1, 2, 0))
        axs[i].set_title(labels[preds[i].item()])
        axs[i].axis('off')

        num_images_to_display -= 1
        if num_images_to_display == 0:
            break

    plt.tight_layout()
    plt.show()
    
    
## debug "Predict" function & "View_Predict_result" function
#test_dir = os.path.join(data_dir, 'test')
#transform = tsfm.Compose([
#    tsfm.Resize((224, 224)),
#    tsfm.ToTensor(),
#])
#test_set = Pred_data(
#    root_dir=test_dir,
#    transform=transform
#)
#model = resnet_50(num_classes=12).cuda()
#
#preds = predict(test_set, model)
#view_pred_result(preds)