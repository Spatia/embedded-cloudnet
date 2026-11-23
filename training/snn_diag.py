import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

SYN_LAYERS = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


def set_v_init(snn, v_init):
    if v_init is None:
        return
    for m in snn.modules():
        if isinstance(m, neuron.BaseNode):
            m.v = float(v_init)


@torch.no_grad()
def snn_forward(snn, x, T, v_init=0.0, burn_in=0):
    functional.reset_net(snn)
    set_v_init(snn, v_init)
    out = 0
    for t in range(T):
        y = snn(x)
        if t >= burn_in:
            out = out + y
    return out / (T - burn_in)


def _record(model, store, accumulate):
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, SYN_LAYERS):
            def hook(mod, inp, name=name):
                x = inp[0].detach()
                if accumulate and name in store:
                    store[name] = store[name] + x
                else:
                    store[name] = x.clone()
            handles.append(m.register_forward_pre_hook(hook))
    return handles


@torch.no_grad()
def layer_errors(ann, snn, x, T, v_init=0.0, burn_in=0):
    a_in, s_in = {}, {}
    hs = _record(ann, a_in, accumulate=False)
    a_out = ann(x)
    for h in hs:
        h.remove()

    functional.reset_net(snn)
    set_v_init(snn, v_init)
    hs = _record(snn, s_in, accumulate=True)
    s_out = 0
    for t in range(T):
        if t == burn_in:
            s_in.clear()
        y = snn(x)
        if t >= burn_in:
            s_out = s_out + y
    for h in hs:
        h.remove()
    n = T - burn_in
    s_out = s_out / n

    rows = []
    for name, a in a_in.items():
        if name not in s_in:
            continue
        s = s_in[name] / n
        err = ((s - a).norm() / a.norm().clamp_min(1e-12)).item()
        bias = ((s.mean() - a.mean()) / a.abs().mean().clamp_min(1e-12)).item()
        rows.append((name, err, bias))
    err = ((s_out - a_out).norm() / a_out.norm().clamp_min(1e-12)).item()
    bias = ((s_out.mean() - a_out.mean()) / a_out.abs().mean().clamp_min(1e-12)).item()
    rows.append(("SORTIE (logits)", err, bias))
    return rows


def print_layer_errors(ann, snn, xs, T, v_init=0.0, burn_in=0):
    acc = {}
    for x in xs:
        for name, e, b in layer_errors(ann, snn, x, T, v_init, burn_in):
            acc.setdefault(name, []).append((e, b))
    print(f"\nT={T}  v_init={v_init}  burn_in={burn_in}")
    print(f"{'Entrée de la couche':<36}{'err_rel':>10}{'biais':>10}")
    for name, v in acc.items():
        e = sum(p[0] for p in v) / len(v)
        b = sum(p[1] for p in v) / len(v)
        print(f"{name:<36}{e:>10.3f}{b:>+10.2%}")