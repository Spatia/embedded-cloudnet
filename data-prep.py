global_dir = "/Users/ck192/Downloads/archive/38-Cloud_training"

red_dir = f"{global_dir}/train_red"
green_dir = f"{global_dir}/train_green"
blue_dir = f"{global_dir}/train_blue"

result_dir = f"{global_dir}/train_rgb"

# Colored images are stored as black and white images in separate folders for each color channel. This function merges the three channels into a single RGB image and saves it to the result directory.

import os
import numpy as np
from PIL import Image

def merge_rgb(red_dir, green_dir, blue_dir, result_dir):
    red_images = sorted(os.listdir(red_dir))
    green_images = sorted(os.listdir(green_dir))
    blue_images = sorted(os.listdir(blue_dir))

    for red_img, green_img, blue_img in zip(red_images, green_images, blue_images):
        red_path = os.path.join(red_dir, red_img)
        green_path = os.path.join(green_dir, green_img)
        blue_path = os.path.join(blue_dir, blue_img)

        # Charger les images TIFF en tant que numpy arrays
        red_data = np.array(Image.open(red_path))
        green_data = np.array(Image.open(green_path))
        blue_data = np.array(Image.open(blue_path))

        # Normaliser en 8-bit (0-255) si nécessaire
        if red_data.dtype == np.uint16:  # 16-bit
            red_data = (red_data / 256).astype(np.uint8)
            green_data = (green_data / 256).astype(np.uint8)
            blue_data = (blue_data / 256).astype(np.uint8)
        elif red_data.dtype == np.float32 or red_data.dtype == np.float64:
            red_data = (red_data * 255).astype(np.uint8)
            green_data = (green_data * 255).astype(np.uint8)
            blue_data = (blue_data * 255).astype(np.uint8)

        # Créer l'image RGB
        rgb_array = np.stack([red_data, green_data, blue_data], axis=2)
        rgb_image = Image.fromarray(rgb_array, mode='RGB')

        result_path = os.path.join(result_dir, red_img)
        rgb_image.save(result_path)
        print(f"Fusionné: {red_img} (dtype={red_data.dtype}, min={red_data.min()}, max={red_data.max()})")

merge_rgb(red_dir, green_dir, blue_dir, result_dir)