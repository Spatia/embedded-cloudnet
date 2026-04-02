import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import warnings

from cloud_dataset import CloudDataset
from unet import Unet_1M, Unet_7M, Unet_31M, Unet_1M_Q

warnings.filterwarnings("ignore", message="The given buffer is not writable")
torch.backends.quantized.engine = 'fbgemm'

def single_image_inference(image_paths, model_pth, device, output_path="inference_result.png"):
    """
    image_paths: dict ou liste avec les 4 fichiers TIFF
    Ex: {'red': 'path/red.tif', 'green': 'path/green.tif', 'nir': 'path/nir.tif', 'blue': 'path/blue.tif'}
    output_path: path to save the result figure (default: inference_result.png)
    """
    if model_pth.endswith(".pt"):
        device = "cpu"
        model = torch.jit.load(model_pth, map_location=device)
    elif model_pth.endswith(".pt2"):
        loaded_program = torch.export.load(model_pth)
        model = loaded_program.module()
    elif model_pth.__contains__("QAT_FS"):
        device = "cpu"
        model_fp32 = Unet_1M_Q(in_channels=4, num_classes=1).to(device)
        model_fp32.eval()

        import torch.ao.quantization as quantization
        import torch.nn as nn
        import unet_parts
        
        model_fp32.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        per_tensor_qconfig = quantization.QConfig(
            activation=model_fp32.qconfig.activation,
            weight=quantization.default_weight_fake_quant
        )
        for module in model_fp32.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.qconfig = per_tensor_qconfig
                
        # for module in model_fp32.modules():
        #     if isinstance(module, unet_parts.DoubleConv) or hasattr(module, 'fuse_model'):
        #         module.fuse_model()

        model_prepared = quantization.prepare(model_fp32)
        model = quantization.convert(model_prepared)

        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.__contains__("QAT_FT"):
        device = "cpu"
        model_q = Unet_1M_Q(in_channels=4, num_classes=1).to(device)
        model_q.eval()
        torch.backends.quantized.engine = 'fbgemm'

        import torch.ao.quantization as quantization
        import torch.nn as nn
        import unet_parts
        
        model_q.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        per_tensor_qconfig = quantization.QConfig(
            activation=model_q.qconfig.activation,
            weight=quantization.default_weight_fake_quant
        )
        for module in model_q.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.qconfig = per_tensor_qconfig
                
        for module in model_q.modules():
            if isinstance(module, unet_parts.DoubleConv_Q):
                module.fuse_model()
        
        model_q.train() 
        model_prepared = quantization.prepare_qat(model_q)
        model_prepared.eval()
        model = quantization.convert(model_prepared)

        model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        model.eval()


    channels = []
    for band in ['red','green', 'blue',  'nir']:
        img_array = tifffile.imread(image_paths[band]).astype(np.float32)

        img_array = img_array / 65535.0
        channels.append(img_array)
    
    # 4 canaux
    img_4ch = np.stack(channels, axis=0) 
    img_tensor = torch.from_numpy(img_4ch).to(device).unsqueeze(0)
    
    # Inférence
    if not model_pth.endswith(".pt2"):
        model.eval()

    with torch.no_grad():
        pred_mask = model(img_tensor)
    
    # Post-traitement
    pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
    print(f"Min logit: {pred_mask.min():.4f}")
    print(f"Max logit: {pred_mask.max():.4f}")
    print(f"Mean logit: {pred_mask.mean():.4f}")
    pred_mask = np.where(pred_mask >2, 1, 0)  # Thresholding
    
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
    if 'mask' in image_paths and image_paths['mask'] is not None:
        axes[1, 2].imshow(tifffile.imread(image_paths['mask']), cmap="gray")
        axes[1, 2].set_title("Masque de référence")
    else:
        axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()

def batch_comparison_inference(image_paths_list, models, device, output_path="inference_result.png", model_names=None):
    """
    image_paths_list: list of dicts, each containing paths to 4-channel images
    models: list of 2 model paths to compare
    device: torch device
    output_path: path to save the result figure
    model_names: list of names for each model (optional)
    """
    num_images = len(image_paths_list)
    fig, axes = plt.subplots(len(models), num_images, figsize=(3*num_images, 10))
    
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    # Load models
    loaded_models = []
    for model_pth in models:
        if model_pth.endswith(".pt"):
            device = "cpu"
            model = torch.jit.load(model_pth, map_location=device)
        elif model_pth.endswith(".pt2"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loaded_program = torch.export.load(model_pth)
            model = loaded_program.module()
        elif model_pth.endswith("unet_1M.pth"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet_1M(in_channels=4, num_classes=1).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("QAT_FS"):
            device = "cpu"
            model_fp32 = Unet_1M_Q(in_channels=4, num_classes=1).to(device)
            model_fp32.eval()

            import torch.ao.quantization as quantization
            import torch.nn as nn
            import unet_parts
            
            model_fp32.qconfig = quantization.get_default_qat_qconfig('fbgemm')
            per_tensor_qconfig = quantization.QConfig(
                activation=model_fp32.qconfig.activation,
                weight=quantization.default_weight_fake_quant
            )
            for module in model_fp32.modules():
                if isinstance(module, nn.ConvTranspose2d):
                    module.qconfig = per_tensor_qconfig

            model_prepared = quantization.prepare(model_fp32)
            model = quantization.convert(model_prepared)

            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("QAT_FT"):
            device = "cpu"
            model_q = Unet_1M_Q(in_channels=4, num_classes=1).to(device)
            model_q.eval()
            torch.backends.quantized.engine = 'fbgemm'

            import torch.ao.quantization as quantization
            import torch.nn as nn
            import unet_parts
            
            model_q.qconfig = quantization.get_default_qat_qconfig('fbgemm')
            per_tensor_qconfig = quantization.QConfig(
                activation=model_q.qconfig.activation,
                weight=quantization.default_weight_fake_quant
            )
            for module in model_q.modules():
                if isinstance(module, nn.ConvTranspose2d):
                    module.qconfig = per_tensor_qconfig
                    
            for module in model_q.modules():
                if isinstance(module, unet_parts.DoubleConv_Q):
                    module.fuse_model()
            
            model_q.train() 
            model_prepared = quantization.prepare_qat(model_q)
            model_prepared.eval()
            model = quantization.convert(model_prepared)

            model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
            model.eval()
        
        if not model_pth.endswith(".pt2"):
            model.eval()
        loaded_models.append((model, model_pth))

    # Process each image
    for img_idx, image_paths in enumerate(image_paths_list):
        channels = []
        for band in ['red', 'green', 'blue', 'nir']:
            img_array = tifffile.imread(image_paths[band]).astype(np.float32)
            img_array = img_array / 65535.0
            channels.append(img_array)
        
        img_4ch = np.stack(channels, axis=0)
        img_tensor_base = torch.from_numpy(img_4ch).unsqueeze(0)
        
        # Inference for each model
        for model_idx, (model, model_pth) in enumerate(loaded_models):
            current_device = next(model.parameters()).device if model_pth.endswith(".pth") else "cpu"
            img_tensor = img_tensor_base.to(current_device)
            
            with torch.no_grad():
                pred_mask = model(img_tensor)
            
            pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
            pred_mask = np.where(pred_mask > 2, 1, 0)
            
            axes[model_idx, img_idx].imshow(pred_mask[0], cmap="gray")
            if model_names:
                axes[model_idx, img_idx].set_title(f"{model_names[model_idx]} - Image {img_idx+1}")
            else:
                axes[model_idx, img_idx].set_title(f"Model {model_idx+1} - Image {img_idx+1}")
            axes[model_idx, img_idx].set_xticks([])
            axes[model_idx, img_idx].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Comparison figure saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    image_names = [("92_5_by_8_LC08_L1TP_029041_20160720_20170222_01_T1", "test"), ("49_3_by_9_LC08_L1TP_032035_20160420_20170223_01_T1", "test"), ("148_7_by_16_LC08_L1TP_061017_20160720_20170223_01_T1", "train")]
    image_list = []
    for image_name, type in image_names:
        image_list.append({
            'red': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_red/red_patch_{image_name}.TIF',
            'green': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_green/green_patch_{image_name}.TIF',
            'blue': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_blue/blue_patch_{image_name}.TIF',
            'nir': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_nir/nir_patch_{image_name}.TIF',
            'mask': f'./dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_gt/gt_patch_{image_name}.TIF' if type == "train" else None
        })

    print(image_list)


    model_pth = "unet_1M_QAT_FT_int8.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The two first crop the mask
# single_image_inference(image_list[0], model_pth, device, output_path="test_1.png")
# single_image_inference(image_list[1], model_pth, device, output_path="test_2.png")
# single_image_inference(image_list[2], model_pth, device, output_path="train.png")

    batch_comparison_inference(image_list, ["./models/unet_1M.pth", "./models/unet_1M_PTQ_int8.pt", "./models/unet_1M_QAT_FS_int8.pth","./models/unet_1M_QAT_FT_int8.pth"], device, output_path="comparison_all.png", model_names=["1M FP32","1M PTQ INT8","1M QAT FS INT8","1M QAT FT INT8"])