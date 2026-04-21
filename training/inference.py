import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile
import warnings
import torchprofile
import time

from unet import Unet, Unet_31M, Unet_1M_Q, Unet_Depthwise

matplotlib.use('Agg')
warnings.filterwarnings("ignore", message="The given buffer is not writable")
torch.backends.quantized.engine = 'qnnpack'

class ONNXModelWrapper:
    def __init__(self, model_path):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        import torch

        x_np = x.cpu().numpy()
        # Inférence
        outputs = self.session.run(None, {self.input_name: x_np})
        return torch.from_numpy(outputs[0]).to(x.device)

    def eval(self):
        # Ne pas déclencher d'erreur lors de l'appel à model.eval()
        pass

def single_image_inference(image_paths, model_pth, device, output_path="inference_result.png", threshold=0.5, benchmark=False):
    """
    image_paths: dict ou liste avec les 4 fichiers TIFF
    Ex: {'red': 'path/red.tif', 'green': 'path/green.tif', 'nir': 'path/nir.tif', 'blue': 'path/blue.tif'}
    output_path: path to save the result figure (default: inference_result.png)
    threshold: threshold for binarizing the prediction (default: 0.5)
    """
    start_time = time.time()
    if model_pth.endswith(".pt"):
        device = "cpu"
        model = torch.jit.load(model_pth, map_location=device)
    elif model_pth.endswith(".pt2"):
        loaded_program = torch.export.load(model_pth)
        model = loaded_program.module()
    elif model_pth.endswith(".onnx"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ONNXModelWrapper(model_pth)
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
    elif model_pth.__contains__("unet_dw_96k_aspp"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32, dilation_rates=[1, 2, 4]).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.__contains__("unet_dw_96k"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.__contains__("unet_400k"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet(in_channels=4, num_classes=1, down_layers=1, up_layers=1, first_layer_channel=64).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.__contains__("unet_460k"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.__contains__("unet_1M"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=64).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.__contains__("unet_7M"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet(in_channels=4, num_classes=1, down_layers=3, up_layers=3, first_layer_channel=64).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.__contains__("unet_31M"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet_31M(in_channels=4, num_classes=1).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    elif model_pth.endswith(".onnx"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ONNXModelWrapper(model_pth)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unet(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=64).to(device)
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
    if not model_pth.endswith(".pt2"):
        model.eval()
    
    if model_pth.__contains__("ds_"):
        # récupérer la taille d'entrée attendue par le modèle à partir de son nom
        import re
        match = re.search(r"ds_(\d+)", model_pth)
        if match:
            target_size = int(match.group(1))
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            print(f"Warning: Could not determine target size from model name {model_pth}. Using original size.")

    with torch.no_grad():
        pred_mask = model(img_tensor)
    
    # Post-traitement
    pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
    if not benchmark:
        print(f"Min logit: {pred_mask.min():.4f}")
        print(f"Max logit: {pred_mask.max():.4f}")
        print(f"Mean logit: {pred_mask.mean():.4f}")
    pred_mask = np.where(pred_mask > threshold, 1, 0)  # Thresholding

    if model_pth.__contains__("ds_"):
        pred_mask = torch.from_numpy(pred_mask).float().unsqueeze(0)
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=(384, 384), mode='nearest')
        pred_mask = pred_mask.squeeze(0).cpu().numpy()
    
    end_time = time.time()

    if benchmark:
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Inference time: {inference_time:.2f} ms")
        return
    
    # Affichage
    if not benchmark:
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

def batch_comparison_inference(image_paths_list, models, thresholds, device, output_path="inference_result.png", model_names=None):
    """
    image_paths_list: list of dicts, each containing paths to 4-channel images
    models: list of 2 model paths to compare
    thresholds: list of thresholds for each model to binarize the output (same order as models)
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
        elif model_pth.endswith(".onnx"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ONNXModelWrapper(model_pth)
        elif model_pth.__contains__("unet_dw_96k_aspp"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32, dilation_rates=[1, 2, 4]).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("unet_dw_96k"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("unet_400k"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet(in_channels=4, num_classes=1, down_layers=1, up_layers=1, first_layer_channel=64).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("unet_460k"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("unet_1M"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=64).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("unet_7M"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet(in_channels=4, num_classes=1, down_layers=3, up_layers=3, first_layer_channel=64).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.__contains__("unet_31M"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet_31M(in_channels=4, num_classes=1).to(device)
            model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        elif model_pth.endswith(".onnx"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ONNXModelWrapper(model_pth)
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

    source_pred_masks = []
    IoUs = [[] for _ in range(len(models))]
    macs = [[] for _ in range(len(models))]
    logits = [[] for _ in range(len(models))]

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

            if model_pth.__contains__("ds_"):
                # récupérer la taille d'entrée attendue par le modèle à partir de son nom
                import re
                match = re.search(r"ds_(\d+)", model_pth)
                if match:
                    target_size = int(match.group(1))
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
                else:
                    print(f"Warning: Could not determine target size from model name {model_pth}. Using original size.")
            
            with torch.no_grad():
                try:
                    macs[model_idx].append(torchprofile.profile_macs(model, img_tensor))
                except Exception as _:
                    macs[model_idx].append(float('nan'))
                pred_mask = model(img_tensor)
                logits[model_idx].append([pred_mask.min().item(), pred_mask.max().item(), pred_mask.mean().item()])
            
            pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
            pred_mask = np.where(pred_mask > thresholds[model_idx], 1, 0)
            if model_pth.__contains__("ds_"):
                pred_mask = torch.from_numpy(pred_mask).float().unsqueeze(0)
                pred_mask = torch.nn.functional.interpolate(pred_mask, size=(384, 384), mode='nearest')
                pred_mask = pred_mask.squeeze(0).cpu().numpy()
                
            if model_idx == 0:
                source_pred_masks.append(pred_mask)
                IoUs[model_idx].append(1)
            else:
                iou = calculate_iou(pred_mask, source_pred_masks[img_idx])
                IoUs[model_idx].append(iou)

            axes[model_idx, img_idx].imshow(pred_mask[0], cmap="gray")
            if model_names:
                title_suffix = f"IoU = {IoUs[model_idx][-1]:.4f}" if model_idx > 0 else "Reference"
                axes[model_idx, img_idx].set_title(f"{model_names[model_idx]} - {title_suffix}")
            else:
                axes[model_idx, img_idx].set_title(f"Model {model_idx+1} - Image {img_idx+1}")
            axes[model_idx, img_idx].set_xticks([])
            axes[model_idx, img_idx].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Comparison figure saved to {output_path}")
    for model_idx, (model, model_pth) in enumerate(loaded_models):
            avg_iou = np.mean(IoUs[model_idx])
            avg_macs = np.mean(macs[model_idx])
            logits_arr = np.asarray(logits[model_idx])
            print(f"Model {model_idx+1} ({model_pth})   \t IoU = {avg_iou:.4f}\t MACs = {avg_macs:.2f}\t Logits (min, max, mean) = {logits_arr[:,0].mean():.4f}, {logits_arr[:,1].mean():.4f}, {logits_arr[:,2].mean():.4f}")
    plt.close()

def calculate_iou(mask1, mask2):
    mask1 = np.ndarray.flatten(mask1)
    mask2 = np.ndarray.flatten(mask2)

    m1 = np.asarray(mask1).astype(bool)
    m2 = np.asarray(mask2).astype(bool)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0
        
    iou = intersection / union
    return float(iou)

if __name__ == "__main__":
    image_names = [("92_5_by_8_LC08_L1TP_029041_20160720_20170222_01_T1", "test"), ("49_3_by_9_LC08_L1TP_032035_20160420_20170223_01_T1", "test"), ("148_7_by_16_LC08_L1TP_061017_20160720_20170223_01_T1", "train")]
    image_list = []
    for image_name, type in image_names:
        image_list.append({
            'red': f'../dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_red/red_patch_{image_name}.TIF',
            'green': f'../dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_green/green_patch_{image_name}.TIF',
            'blue': f'../dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_blue/blue_patch_{image_name}.TIF',
            'nir': f'../dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_nir/nir_patch_{image_name}.TIF',
            'mask': f'../dataset/38-Cloud_{"training" if type == "train" else "test"}/{type}_gt/gt_patch_{image_name}.TIF' if type == "train" else None
        })


    model_pth = "../models/unet_dw_96k_ds_192.pth"
    threshold = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The two first crop the mask
    single_image_inference(image_list[0], model_pth, device, output_path="test_1.png", threshold=threshold)
    #single_image_inference(image_list[1], model_pth, device, output_path="test_2.png", threshold=threshold)
    #single_image_inference(image_list[2], model_pth, device, output_path="train.png", threshold=threshold)

    # batch_comparison_inference(image_list, ["./models/unet_31M.pth", "./models/unet_7M.pth", "./models/unet_1M.pth", "unet_1M.pth", "./models/unet_400k.pth"], [0,2,2,2,3], device, output_path="comparison_all.png", model_names=["31M (4D4U)","7M (3D3U)","1M (2D2U)", "1M V2 (2D2U)", "400k (1D1U)"])
    # batch_comparison_inference(image_list, ["./models/unet_31M.pth", "./models/unet_1M.pth", "./models/unet_460k.pth", "./models/unet_400k.pth"], [0,2,3,3], device, output_path="comparison_all_2.png", model_names=["31M (64FF)","1M (64FF)","460k (32FF)", "400k (64FF)"])
    # batch_comparison_inference(image_list, ["./models/unet_1M.pth", "./models/unet_1M_PTQ_int8.pt", "./models/unet_1M_QAT_FS_int8.pth","./models/unet_1M_QAT_FT_int8.pth"], [2,2,2,2], device, output_path="comparison_all_Q.png", model_names=["1M FP32","1M PTQ INT8","1M QAT FS INT8","1M QAT FT INT8"])
    # batch_comparison_inference(image_list, ["./models/unet_1M.pth", "./models/unet_1M_PTQ_int8.pt", "./models/unet_dw_96k_int8.pt", "unet_dw_96k_ds_96_save.pth"], [2,2,2,0], device, output_path="comparison_all.png", model_names=["1M FP32","1M PTQ INT8", "96k INT8", "96k DS 96 FP32"])