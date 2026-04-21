import torch
from unet import Unet, Unet_Depthwise, Unet_31M
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat
import numpy as np
import tifffile
import os


def export_model_to_onnx(model_path, output_path):
    model = Unet_31M(in_channels=4, num_classes=1)
    # model = Unet(
    #     in_channels=4, num_classes=1, down_layers=3, up_layers=3, first_layer_channel=64
    # )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x = torch.randn(1, 4, 384, 384)

    torch.onnx.export(
        model,
        x,
        output_path,
        input_names=["input"],
        output_names=["mask"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch"},
            "mask": {0: "batch"},
        },
    )

def onnx_PTQ(model_input, model_output, calibration_data):
    class CalibReader(CalibrationDataReader):
        def __init__(self):
            self.data = iter([
                {"input": img.numpy()} for img in calibration_data
            ])
        def get_next(self):
            return next(self.data, None)

    quantize_static(
        model_input=model_input,
        model_output=model_output,
        calibration_data_reader=CalibReader(),
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
    )

if __name__ == "__main__":
    original_model_path = '../models/unet_31M.pth'
    onnx_out_path = '../models/unet_31M.onnx'
    quantized_onnx_out_path = '../models/unet_31M_int8.onnx'

    image_folder = "../dataset/38-Cloud_training/train_"

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

    export_model_to_onnx(original_model_path, onnx_out_path)
    onnx_PTQ(onnx_out_path, quantized_onnx_out_path, calibration_data)