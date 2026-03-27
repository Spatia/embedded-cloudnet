import os
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms

class CloudDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        self.images_red = sorted([root_path+"/38-Cloud_training/train_red/" + i for i in os.listdir(root_path+"/38-Cloud_training/train_red/")])
        self.images_green = sorted([root_path+"/38-Cloud_training/train_green/" + i for i in os.listdir(root_path+"/38-Cloud_training/train_green/")])
        self.images_blue = sorted([root_path+"/38-Cloud_training/train_blue/" + i for i in os.listdir(root_path+"/38-Cloud_training/train_blue/")])
        self.images_nir = sorted([root_path+"/38-Cloud_training/train_nir/" + i for i in os.listdir(root_path+"/38-Cloud_training/train_nir/")])
        self.masks = sorted([root_path+"/38-Cloud_training/train_gt/" + i for i in os.listdir(root_path+"/38-Cloud_training/train_gt/")])

    def __getitem__(self, index):
        img_red = np.array(Image.open(self.images_red[index]), dtype=np.float32)
        img_green = np.array(Image.open(self.images_green[index]), dtype=np.float32)
        img_blue = np.array(Image.open(self.images_blue[index]), dtype=np.float32)
        img_nir = np.array(Image.open(self.images_nir[index]), dtype=np.float32)

        # Normalisation
        img_red = img_red / 65535.0
        img_green = img_green / 65535.0
        img_blue = img_blue / 65535.0
        img_nir = img_nir / 65535.0

        img_stacked = np.stack([img_red, img_green, img_blue, img_nir], axis=0)

        mask = np.array(Image.open(self.masks[index]).convert("L"), dtype=np.float32)
        mask = mask / 255.0
        
        return torch.from_numpy(img_stacked), torch.from_numpy(mask).unsqueeze(0)

    def __len__(self):
        return len(self.images_red)