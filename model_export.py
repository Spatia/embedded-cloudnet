import torch
import torch.export
from unet import Unet_1M

model = Unet_1M(in_channels=4, num_classes=1)
model.load_state_dict(torch.load('unet.pth'))
model.eval()

example_input = (torch.randn(1, 4, 384, 384),) 
exported_program = torch.export.export(model, example_input)

torch.export.save(exported_program, 'unet_exported.pte')