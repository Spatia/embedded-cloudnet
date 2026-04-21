import torch
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping
import torch
import torch.export
from unet import Unet, Unet_Depthwise
import numpy as np
import tifffile
import os

def quantize_cnn_int8(model_path, output_path, calibration_data):
    """
    Quantification statique INT8 pour les CNNs (comme U-Net).
    Réduit la taille par 4 et accélère l'inférence sur CPU/Edge.
    """
    print(f"Loading model from {model_path} for Native INT8 Quantization...")
    
    model = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    
    example_inputs = (torch.randn(1, 4, 384, 384),) 
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)

    with torch.no_grad():
        for batch in calibration_data:
            prepared_model(batch)

    quantized_model = convert_fx(prepared_model)

    quantized_model.eval()

    example_input = (torch.randn(1, 4, 384, 384),) 
    traced_model = torch.jit.trace(quantized_model, example_input)
    traced_model.save(output_path)
    print(f"Quantized model saved to {output_path}")

if __name__ == "__main__":
    original_model_path = './models/unet_dw_96k.pth'
    quantized_out_path = './models/unet_dw_96k_int8.pt'

    image_folder = "./dataset/38-Cloud_training/train_"

    images_arrays = []

    raw_images = [[], [], [], []]
    for file in os.listdir(image_folder + "red/")[:100]:
        raw_images[0].append(tifffile.imread(os.path.join(image_folder + "red/", file)).astype(np.float32) / 65535.0)
    for file in os.listdir(image_folder + "green/")[:100]:
        raw_images[1].append(tifffile.imread(os.path.join(image_folder + "green/", file)).astype(np.float32) / 65535.0)
    for file in os.listdir(image_folder + "blue/")[:100]:
        raw_images[2].append(tifffile.imread(os.path.join(image_folder + "blue/", file)).astype(np.float32) / 65535.0)
    for file in os.listdir(image_folder + "nir/")[:100]:
        raw_images[3].append(tifffile.imread(os.path.join(image_folder + "nir/", file)).astype(np.float32) / 65535.0)
    
    for i in range(len(raw_images[0])):
        img_red = raw_images[0][i]
        img_green = raw_images[1][i]
        img_blue = raw_images[2][i]
        img_nir = raw_images[3][i]
        img_4ch = np.stack([img_red, img_green, img_blue, img_nir], axis=0)
        images_arrays.append(torch.from_numpy(img_4ch).unsqueeze(0))
    
    calibration_data = [img.to("cpu") for img in images_arrays] 
    
    quantize_cnn_int8(original_model_path, quantized_out_path, calibration_data)