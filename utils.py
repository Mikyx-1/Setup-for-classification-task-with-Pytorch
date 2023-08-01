### Initialise Dataset, Train Transform, Val Transform, Training functions, Validation Functions 

import torch
import cv2
import glob
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


class Classification_Dataset(Dataset):                                   
    def __init__(self, folder, transform = None):
        self.img_dirs = glob.glob(folder + "/*/*")
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_dir = self.img_dirs[idx]
        label = self.get_label(img_dir)

        image = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
        return image, label
    
    def get_label(self, img_directory):
        root, name = os.path.split(img_directory)
        root, label = os.path.split(root)
        return float(label)
    


def validator(model, test_loader, device):
    num_samples = 0
    num_corrects = 0
    model = model.to(device)
    model.eval()
    print("Evaluating....")
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().to(device)
            labels = labels.float().to(device)
            preds = model(data).max(1)[1]
            num_samples += labels.size(0)
            num_corrects += len(preds[preds == labels])

    model.train()
    return num_corrects/num_samples


def train(model, train_loader, loss_fn, optimizer, num_epochs, device, validator = None, 
          test_loader = None, save_weights = True):
    model = model.to(device)
    for epoch in range(num_epochs):
        for data, labels in tqdm(train_loader):
            data = data.float().to(device)
            labels = labels.long().to(device)
            preds = model(data).to(device)
            optimizer.zero_grad()
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 and validator is not None and test_loader is not None:
            validation_acc = validator(model, test_loader, device)
            print(f"Epoch {epoch}   Loss: {loss.item()}  Val Acc: {validation_acc}")
        else:
            print(f"Epoch {epoch}   Loss: {loss.item()}")

    if save_weights == True:
        torch.save(model.state_dict(), "./My_model.pt")



def visualise_data(dataset, classes):  
    ### classes in format {label: name_label}
    _, axes = plt.subplots(4, 4, figsize = (10, 5))
    for i in range(16):
        image, label = dataset[random.randint(0, len(dataset) - 2)]
        if image.shape[0] < image.shape[-1] and len(image.shape) == 3:  ### image is in Torch format
            image = torch.einsum("cab->abc", image)
            axes[i//4, i%4].imshow(image)
            axes[i//4, i%4].set_title(classes[label])
            axes[i//4, i%4].axis("off")

        if len(image.shape) == 3 and image.shape[0] > image.shape[-1]:
            axes[i//4, i%4].imshow(image)
            axes[i//4, i%4].set_title(classes[label])
            axes[i//4, i%4].axis("off")
            









