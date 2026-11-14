import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class BaselineUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.name = "Baseline_UNet"

        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        b = self.bottleneck(p3)

        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.head(d1)
