import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, functional


class SpikingMaxPool2d(base.MemoryModule):
    def __init__(self, pool: nn.MaxPool2d):
        super().__init__()
        self.kernel_size, self.stride = pool.kernel_size, pool.stride
        self.padding, self.dilation = pool.padding, pool.dilation
        self.ceil_mode = pool.ceil_mode
        self.register_memory("count", 0.0)

    def forward(self, x):
        if not isinstance(self.count, torch.Tensor):
            self.count = torch.zeros_like(x)
        self.count = self.count + x
        _, idx = F.max_pool2d(self.count, self.kernel_size, self.stride, self.padding,
                              self.dilation, self.ceil_mode, return_indices=True)
        out = x.flatten(2).gather(2, idx.flatten(2))
        return out.view(idx.shape)


def replace_maxpool(model: nn.Module) -> int:
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.MaxPool2d)]
    for name, m in targets:
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, SpikingMaxPool2d(m))
    functional.invalidate_reset_cache(model)
    return len(targets)
