import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

from unet_parts import DepthwiseSeparableConv, DoubleConv_Depthwise, DownSample, DownSample_Depthwise, DownSample_Q, UpSample, UpSample_Depthwise, UpSample_Q, DoubleConv, DoubleConv_Q, DownSample_31M, UpSample_31M, DoubleConv_31M

class Unet_31M(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down1 = DownSample_31M(in_channels, 64)
        self.down2 = DownSample_31M(64, 128)
        self.down3 = DownSample_31M(128, 256)
        self.down4 = DownSample_31M(256, 512)

        self.bottleneck = DoubleConv_31M(512, 1024)

        self.up1 = UpSample_31M(1024, 512)
        self.up2 = UpSample_31M(512, 256)
        self.up3 = UpSample_31M(256, 128)
        self.up4 = UpSample_31M(128, 64)

        self.out = nn.Conv2d(64, out_channels=num_classes, kernel_size=1)
    
    def forward(self, x):
        down_1, p1 = self.down1(x)
        down_2, p2 = self.down2(p1)
        down_3, p3 = self.down3(p2)
        down_4, p4 = self.down4(p3)

        b=self.bottleneck(p4)

        up_1 = self.up1(b, down_4)
        up_2 = self.up2(up_1, down_3)
        up_3 = self.up3(up_2, down_2)
        up_4 = self.up4(up_3, down_1)

        out=self.out(up_4)

        return out
    
class Unet(nn.Module):
    def __init__(self, in_channels, num_classes, down_layers=3, up_layers=3, first_layer_channel=64):
        super().__init__()
        self.down_layers = down_layers
        self.up_layers = up_layers

        channel=first_layer_channel

        self.down1 = DownSample(in_channels, channel)
        for i in range(2, down_layers+1):
            setattr(self, f"down{i}", DownSample(channel, channel*2))
            channel*=2

        self.bottleneck = DoubleConv(channel, channel*2)

        for i in range(1, up_layers+1):
            setattr(self, f"up{i}", UpSample(channel*2, channel))
            channel//=2

        self.out = nn.Conv2d(channel*2, out_channels=num_classes, kernel_size=1)
    
    def forward(self, x):
        downs = []

        for i in range(1, self.down_layers+1):
            down, x = getattr(self, f"down{i}")(x)
            downs.append(down)

        b=self.bottleneck(x)

        for i in range(1, self.up_layers+1):
            up = getattr(self, f"up{i}")
            b = up(b, downs[-i])

        out=self.out(b)

        return out
    
class Unet_Depthwise(nn.Module):
    def __init__(self, in_channels, num_classes, down_layers=3, up_layers=3, first_layer_channel=64):
        super().__init__()
        self.down_layers = down_layers
        self.up_layers = up_layers

        channel=first_layer_channel

        self.down1 = DownSample_Depthwise(in_channels, channel)
        for i in range(2, down_layers+1):
            setattr(self, f"down{i}", DownSample_Depthwise(channel, channel*2))
            channel*=2

        self.bottleneck = DoubleConv_Depthwise(channel, channel*2)

        for i in range(1, up_layers+1):
            setattr(self, f"up{i}", UpSample_Depthwise(channel*2, channel))
            channel//=2

        self.out = DepthwiseSeparableConv(channel*2, out_channels=num_classes, kernel_size=1, padding=0)
    
    def forward(self, x):
        downs = []

        for i in range(1, self.down_layers+1):
            down, x = getattr(self, f"down{i}")(x)
            downs.append(down)

        b=self.bottleneck(x)

        for i in range(1, self.up_layers+1):
            up = getattr(self, f"up{i}")
            b = up(b, downs[-i])

        out=self.out(b)

        return out

class Unet_1M_Q(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.down1 = DownSample_Q(in_channels, 64)
        self.down2 = DownSample_Q(64, 128)

        self.bottleneck = DoubleConv_Q(128, 256)

        self.up1 = UpSample_Q(256, 128)
        self.up2 = UpSample_Q(128, 64)

        self.out = nn.Conv2d(64, out_channels=num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.quant(x)
        down_1, p1 = self.down1(x)
        down_2, p2 = self.down2(p1)

        b=self.bottleneck(p2)

        up_1 = self.up1(b, down_2)
        up_2 = self.up2(up_1, down_1)

        out=self.out(up_2)
        out = self.dequant(out)

        return out

if __name__=='__main__':
    double_conv = DoubleConv(256,256)
    print(double_conv)

    input_image = torch.rand((1,3,384,384))
    model=Unet(3,10)
    output = model(input_image)
    print(output.size())