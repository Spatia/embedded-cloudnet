"""
Standalone inference, only depends on `unet.py` for the architecture.
Heavily relies on `spikingjelly.activation_based.ann2snn`
"""

import torch
import numpy as np
import tifffile
import copy
from spikingjelly.activation_based import ann2snn, functional

from unet import Unet_Depthwise

MODEL_PTH = "../models/unet_dw_145k_aspp_ds_192.pth_final"
INPUT_SIZE = 192
T = 128
THRESHOLD = 0.0


def calculate_iou(mask1, mask2):
    m1, m2 = np.asarray(mask1).astype(bool).ravel(), np.asarray(mask2).astype(bool).ravel()
    union = np.logical_or(m1, m2).sum()
    return 1.0 if union == 0 else float(np.logical_and(m1, m2).sum() / union)


def load_image(name, folder="../dataset/38-Cloud_training/train_"):
    bands = [tifffile.imread(f"{folder}{b}/{b}_{name}.TIF").astype(np.float32) / 65535.0
             for b in ["red", "green", "blue", "nir"]]
    img = torch.from_numpy(np.stack(bands, axis=0)).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    mask = tifffile.imread(f"{folder}gt/gt_{name}.TIF").astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)[None, None]
    mask = torch.nn.functional.interpolate(mask, size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")
    return img, mask.numpy()


@torch.no_grad()
def snn_forward(snn, x, T):
    functional.reset_net(snn)
    out = 0
    for _ in range(T):
        out = out + snn(x)
    return out / T


if __name__ == "__main__":
    names = open("../dataset/training_patches_38-cloud_nonempty.csv").read().split()[1:]
    rng = np.random.default_rng(0)
    rng.shuffle(names)
    calib_names, test_names = names[:100], names[100:120]

    calibration_data = [load_image(n)[0] for n in calib_names]

    ann = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2,
                         first_layer_channel=32, dilation_rates=[1, 2, 4])
    ann.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
    ann.eval()

    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data, mode="99.9%")
    snn = ann2snn.Converter(recipe).convert(copy.deepcopy(ann))
    snn.eval()
    print(snn)

    #ANN vs SNN
    ious_ann, ious_snn, ious_agree = [], [], []
    for n in test_names:
        x, gt = load_image(n)
        with torch.no_grad():
            logit_ann = ann(x)
        logit_snn = snn_forward(snn, x, T)

        m_ann = (logit_ann > THRESHOLD).numpy()
        m_snn = (logit_snn > THRESHOLD).numpy()
        ious_ann.append(calculate_iou(m_ann, gt))
        ious_snn.append(calculate_iou(m_snn, gt))
        ious_agree.append(calculate_iou(m_ann, m_snn))

    print(f"T = {T}")
    print(f"IoU ANN vs GT : {np.mean(ious_ann):.4f}")
    print(f"IoU SNN vs GT : {np.mean(ious_snn):.4f}")
    print(f"IoU ANN vs SNN: {np.mean(ious_agree):.4f}")
