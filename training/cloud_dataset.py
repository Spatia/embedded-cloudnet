import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
import torch

class CloudDataset(Dataset):
    def __init__(self, root_path, csv_path=None, augment=True, resize=None):
        self.root_path = root_path
        self.augment = augment
        self.resize = resize

        # Si un CSV est fourni, on charge uniquement les images listées dedans
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            # Les noms dans le CSV sont du type "patch_X_Y...", les fichiers rajoutent le préfixe de couleur et ".TIF"
            self.images_red = [os.path.join(root_path, "38-Cloud_training/train_red/", f"red_{name}.TIF") for name in df['name']]
            self.images_green = [os.path.join(root_path, "38-Cloud_training/train_green/", f"green_{name}.TIF") for name in df['name']]
            self.images_blue = [os.path.join(root_path, "38-Cloud_training/train_blue/", f"blue_{name}.TIF") for name in df['name']]
            self.images_nir = [os.path.join(root_path, "38-Cloud_training/train_nir/", f"nir_{name}.TIF") for name in df['name']]
            self.masks = [os.path.join(root_path, "38-Cloud_training/train_gt/", f"gt_{name}.TIF") for name in df['name']]
        else:
            # Ancien comportement de secours (toutes les images)
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

        # Rescale les images réduire la charge de calcul
        img_red = Image.fromarray(img_red)
        img_green = Image.fromarray(img_green)
        img_blue = Image.fromarray(img_blue)
        img_nir = Image.fromarray(img_nir)

        if self.resize is not None:
            img_red = img_red.resize(self.resize, Image.Resampling.BILINEAR)
            img_green = img_green.resize(self.resize, Image.Resampling.BILINEAR)
            img_blue = img_blue.resize(self.resize, Image.Resampling.BILINEAR)
            img_nir = img_nir.resize(self.resize, Image.Resampling.BILINEAR)

        img_red = np.array(img_red, dtype=np.float32)
        img_green = np.array(img_green, dtype=np.float32)
        img_blue = np.array(img_blue, dtype=np.float32)
        img_nir = np.array(img_nir, dtype=np.float32)

        img_red = img_red / 65535.0
        img_green = img_green / 65535.0
        img_blue = img_blue / 65535.0
        img_nir = img_nir / 65535.0

        img_stacked = np.stack([img_red, img_green, img_blue, img_nir], axis=0)

        mask = Image.open(self.masks[index]).convert("L")
        # Ratio de nuages avant redimensionnement
        # mask_array = np.array(mask, dtype=np.float32)
        # cloud_ratio = np.mean(mask_array > 0)
        # print(f"Image {index} - Cloud ratio before resizing: {cloud_ratio:.4f}")

        if self.resize is not None:
            mask = mask.resize(self.resize, Image.Resampling.NEAREST)
        mask = np.array(mask, dtype=np.float32)

        # Ratio de nuages après redimensionnement
        # cloud_ratio_resized = np.mean(mask > 0)
        # print(f"Image {index} - Cloud ratio after resizing: {cloud_ratio_resized:.4f}")
        mask = mask / 255.0

        if self.augment:
            k = np.random.randint(0, 4)
            img_stacked = np.rot90(img_stacked, k=k, axes=(1, 2))
            mask = np.rot90(mask, k=k, axes=(0, 1))

            if np.random.rand() < 0.5:
                img_stacked = np.flip(img_stacked, axis=2)
                mask = np.flip(mask, axis=1)

        img_stacked = img_stacked.copy()
        mask = mask.copy()

        return torch.from_numpy(img_stacked), torch.from_numpy(mask).unsqueeze(0)

    def __len__(self):
        return len(self.images_red)