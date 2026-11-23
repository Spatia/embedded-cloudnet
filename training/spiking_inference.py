"""
Standalone inference, only depends on `unet.py` for the architecture.
Heavily relies on `spikingjelly.activation_based.ann2snn`
"""

import torch
import numpy as np
import tifffile
import copy
from spikingjelly.activation_based import ann2snn, functional
from ops_counter import OpCounter, report
from snn_diag import snn_forward, print_layer_errors, layer_errors
from spiking_maxpool import replace_maxpool

from unet import Unet, Unet_Depthwise

MODEL_PTH = "../models/unet_dw_96k.pth"
INPUT_SIZE = 384
T = 32
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


# @torch.no_grad()
# def snn_forward(snn, x, T):
#     functional.reset_net(snn)
#     out = 0
#     for _ in range(T):
#         out = out + snn(x)
#     return out / T


if __name__ == "__main__":
    names = open("../dataset/training_patches_38-cloud_nonempty.csv").read().split()[1:]
    rng = np.random.default_rng(0)
    rng.shuffle(names)
    calib_names, test_names = names[:100], names[100:120]

    calibration_data = [load_image(n)[0] for n in calib_names]

    ann = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32)
    # ann = Unet(in_channels=4, num_classes=1, down_layers=2, up_layers=2)
    ann.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
    ann.eval()

    # Rate coding limits are clearly shown here.
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data, mode="99.9%")
    snn = ann2snn.Converter(recipe).convert(copy.deepcopy(ann))
    print(replace_maxpool(snn), "MaxPool2d replaced by SpikingMaxPool2d")
    snn.eval()
    x = load_image(test_names[0])[0][..., :128, :128]
    for T in (32, 128, 512, 2048):
        rows = layer_errors(ann, snn, x, T=T, v_init=0.0)
        print(f"T={T:5d}  err couche 2 = {rows[1][1]:.3f}   err logits = {rows[-1][1]:.3f}")

    #ANN vs SNN
    ious_ann, ious_snn, ious_agree = [], [], []
    with OpCounter(ann) as ann_cnt, OpCounter(snn, probe_call=T // 2) as snn_cnt:
        for n in test_names:
            print(f"Processing {n}...")
            x, gt = load_image(n)
            with torch.no_grad():
                logit_ann = ann(x)
            logit_snn = snn_forward(snn, x, T, v_init=0.5)

            m_ann = (logit_ann > THRESHOLD).numpy()
            m_snn = (logit_snn > THRESHOLD).numpy()
            ious_ann.append(calculate_iou(m_ann, gt))
            ious_snn.append(calculate_iou(m_snn, gt))
            ious_agree.append(calculate_iou(m_ann, m_snn))

    print(f"ANN IoU: {np.mean(ious_ann):.4f} ± {np.std(ious_ann):.4f}")
    print(f"SNN IoU: {np.mean(ious_snn):.4f} ± {np.std(ious_snn):.4f}")
    print(f"ANN/SNN agreement IoU: {np.mean(ious_agree):.4f} ± {np.std(ious_agree):.4f}")

    report(ann_cnt, snn_cnt, len(test_names), T)
