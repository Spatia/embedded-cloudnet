import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

from cloud_dataset import CloudDataset
from unet import Unet, Unet_1M, Unet_7M, Unet_31M

def single_image_inference(image_paths, model_pth, device, output_path="inference_result.png"):
    """
    image_paths: dict ou liste avec les 4 fichiers TIFF
    Ex: {'red': 'path/red.tif', 'green': 'path/green.tif', 'nir': 'path/nir.tif', 'blue': 'path/blue.tif'}
    output_path: path to save the result figure (default: inference_result.png)
    """
    model = Unet_31M(in_channels=4, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    channels = []
    for band in ['red','green', 'blue',  'nir']:
        img_array = tifffile.imread(image_paths[band]).astype(np.float32)

        img_array = img_array / 65535.0
        channels.append(img_array)
    
    # 4 canaux
    img_4ch = np.stack(channels, axis=0) 
    img_tensor = torch.from_numpy(img_4ch).to(device).unsqueeze(0)
    
    # Inférence
    model.eval()
    with torch.no_grad():
        pred_mask = model(img_tensor)
    
    # Post-traitement
    pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
    print(f"Min logit: {pred_mask.min():.4f}")
    print(f"Max logit: {pred_mask.max():.4f}")
    print(f"Mean logit: {pred_mask.mean():.4f}")
    pred_mask = np.where(pred_mask > 0, 1, 0)  # Thresholding
    
    # Affichage
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    axes[0, 0].imshow(channels[0], cmap="gray")
    axes[0, 0].set_title("Blue")
    axes[0, 1].imshow(channels[2], cmap="gray")
    axes[0, 1].set_title("Red")
    axes[1, 0].imshow(channels[3], cmap="gray")
    axes[0, 2].imshow(channels[1], cmap="gray")
    axes[0, 2].set_title("Green")
    axes[1, 0].set_title("NIR")
    axes[1, 1].imshow(pred_mask[0], cmap="gray")
    axes[1, 1].set_title("Prédiction")
    if 'mask' in image_paths:
        axes[1, 2].imshow(tifffile.imread(image_paths['mask']), cmap="gray")
        axes[1, 2].set_title("Masque de référence")
    else:
        axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()


# Exemple (produce the figure in the README):
if __name__ == "__main__":
    image_name = "92_5_by_8_LC08_L1TP_029041_20160720_20170222_01_T1"
    image_name = "49_3_by_9_LC08_L1TP_032035_20160420_20170223_01_T1"
    image_name = "148_7_by_16_LC08_L1TP_061017_20160720_20170223_01_T1"
    type = "train"  # "test" ou "train"
    image_paths = {
        'red': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_red/red_patch_{image_name}.TIF',
        'green': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_green/green_patch_{image_name}.TIF',
        'blue': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_blue/blue_patch_{image_name}.TIF',
        'nir': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_nir/nir_patch_{image_name}.TIF',
        'mask': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_gt/gt_patch_{image_name}.TIF'
    }
    model_pth = "unet_31M.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    single_image_inference(image_paths, model_pth, device)

"""
image_paths = {
        'red': './dataset/38-Cloud_test/test_red/red_patch_347_17_by_11_LC08_L1TP_029041_20160720_20170222_01_T1.TIF',
        'green': './dataset/38-Cloud_test/test_green/green_patch_347_17_by_11_LC08_L1TP_029041_20160720_20170222_01_T1.TIF',
        'blue': './dataset/38-Cloud_test/test_blue/blue_patch_347_17_by_11_LC08_L1TP_029041_20160720_20170222_01_T1.TIF',
        'nir': './dataset/38-Cloud_test/test_nir/nir_patch_347_17_by_11_LC08_L1TP_029041_20160720_20170222_01_T1.TIF'
    }
"""