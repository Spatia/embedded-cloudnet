"""
Comptage des opérations synaptiques ANN vs SNN (convertie par SpikingJelly).

- ANN : chaque connexion = 1 MAC (multiplication + accumulation), que l'entrée soit nulle ou non.
- SNN : seule une entrée non nulle (un spike) déclenche des opérations -> SOP (synaptic operations).
        Si l'entrée d'une couche est binaire (spikes), ce sont de simples additions (AC).
        Si elle est analogique (ex. 1re conv qui reçoit l'image à chaque pas de temps), ce sont des MAC.

Le comptage est exact : on convolue le masque des entrées non nulles avec un noyau de 1,
ce qui compte, pour chaque sortie, les entrées actives de son champ récepteur (padding exclu).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SYN_LAYERS = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)

# Énergie par opération, 32 bits flottant, 45 nm (Horowitz, ISSCC 2014) — valeurs usuelles dans la littérature SNN
E_MAC = 4.6e-12
E_AC = 0.9e-12


def _count(m, mask):
    ones = torch.ones_like(m.weight)
    if isinstance(m, nn.Conv2d):
        y = m._conv_forward(mask, ones, None)
    elif isinstance(m, nn.ConvTranspose2d):
        y = F.conv_transpose2d(mask, ones, None, m.stride, m.padding,
                               m.output_padding, m.groups, m.dilation)
    else:
        y = F.linear(mask, ones)
    return y.sum().item()


class OpCounter:
    """Hooks sur Conv2d / ConvTranspose2d / Linear. À utiliser comme context manager."""

    def __init__(self, model, probe_call=None):
        self.model = model
        self.probe_call = probe_call  # appel où l'on teste si l'entrée est binaire (SNN : milieu de la simulation)
        self.stats = {}
        self.handles = []

    def __enter__(self):
        for name, m in self.model.named_modules():
            if isinstance(m, SYN_LAYERS):
                self.stats[name] = dict(dense_per_call=None, dense=0.0, active=0.0,
                                        nnz=0, numel=0, calls=0, levels=None)
                self.handles.append(m.register_forward_pre_hook(self._hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def _hook(self, name):
        @torch.no_grad()
        def fn(m, inp):
            x = inp[0]
            s = self.stats[name]
            if s["dense_per_call"] is None:
                s["dense_per_call"] = _count(m, torch.ones_like(x))
            mask = (x != 0).to(x.dtype)
            s["dense"] += s["dense_per_call"]
            s["active"] += _count(m, mask)
            s["nnz"] += int(mask.sum().item())
            s["numel"] += x.numel()
            if self.probe_call is not None and s["calls"] == self.probe_call:
                s["levels"] = torch.unique(x).numel()
            s["calls"] += 1
        return fn

    def is_spiking(self, name):
        lv = self.stats[name]["levels"]
        # spikes : {0, v} ; concat de 2 sources : {0, v1, v2} ; avg-pool 2x2 de spikes : 5 niveaux
        # (toujours faisable en additions). Image ou upsampling bilinéaire : beaucoup de niveaux -> MAC.
        return lv is not None and lv <= 8


def report(ann_cnt, snn_cnt, n_images, T):
    print("\n" + "=" * 92)
    print(f"{'Couche':<28}{'MAC ANN':>12}{'SOP SNN':>12}{'type':>6}{'entrées≠0':>12}{'SNN/ANN':>10}")
    print("-" * 92)
    ann_mac = ann_active = snn_ac = snn_mac = 0.0
    for name, a in ann_cnt.stats.items():
        s = snn_cnt.stats.get(name)
        if s is None or a["calls"] == 0 or s["calls"] == 0:
            continue
        a_ops = a["dense"] / n_images
        s_ops = s["active"] / n_images
        spiking = snn_cnt.is_spiking(name)
        rate = s["nnz"] / s["numel"]
        ann_mac += a_ops
        ann_active += a["active"] / n_images
        if spiking:
            snn_ac += s_ops
        else:
            snn_mac += s_ops
        print(f"{name:<28}{a_ops:>12.3e}{s_ops:>12.3e}{'AC' if spiking else 'MAC':>6}"
              f"{rate:>11.2%}{s_ops / a_ops:>9.2f}x")
    print("-" * 92)

    snn_ops = snn_ac + snn_mac
    e_ann = ann_mac * E_MAC
    e_snn = snn_ac * E_AC + snn_mac * E_MAC
    print(f"Par image, T = {T}")
    print(f"  ANN  MAC denses                  : {ann_mac:.3e}")
    print(f"  ANN  MAC sur entrées non nulles  : {ann_active:.3e}   (si le matériel saute les zéros de ReLU)")
    print(f"  SNN  opérations synaptiques      : {snn_ops:.3e}   (AC {snn_ac:.3e} + MAC {snn_mac:.3e})")
    print(f"  SNN  SOP moyennes par pas de temps: {snn_ops / T:.3e}")
    print()
    print(f"  Facteur opérations  ANN / SNN    : {ann_mac / snn_ops:.2f}x   "
          f"({'réduction' if snn_ops < ann_mac else 'AUGMENTATION'})")
    print(f"  Facteur énergie     ANN / SNN    : {e_ann / e_snn:.2f}x   "
          f"(ANN {e_ann * 1e6:.1f} µJ, SNN {e_snn * 1e6:.1f} µJ ; MAC {E_MAC * 1e12} pJ, AC {E_AC * 1e12} pJ)")

    # Les SOP croissent ~linéairement avec T : T à partir duquel la SNN coûte autant que l'ANN
    e_per_step = e_snn / T
    print(f"  T d'équilibre (opérations, ~)    : {ann_mac / (snn_ops / T):.1f}")
    print(f"  T d'équilibre (énergie, ~)       : {e_ann / e_per_step:.1f}")
    print("=" * 92)