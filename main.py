import torch
from torch import optim, nn
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader
from utils import Classification_Dataset
from model import myModel



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_folder = ""
val_folder = ""

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
NUM_EPOCHS = 50

train_transform = A.Compose([A.Resize(height = IMG_HEIGHT, width = IMG_WIDTH), 
                            ToTensorV2()])

val_transform = A.Compose([A.Resize(height = IMG_HEIGHT, width = IMG_WIDTH), 
                           ToTensorV2()])


train_ds = Classification_Dataset(train_folder, train_transform)
val_ds = Classification_Dataset(val_folder, val_transform)

train_loader = DataLoader(train_ds, batch_size= BATCH_SIZE, shuffle= True)
val_loader = DataLoader(val_ds, batch_size= BATCH_SIZE, shuffle = False)

model = myModel()

####loss fn and Optimizer  ###






##################################################






