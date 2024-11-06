import functools
import os.path

import torch
from torch import nn
from torchvision import models


class Sia_sub_00a(nn.Module):
    def __init__(self, in_channels):
        super(Sia_sub_00a, self).__init__()
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)

        self.pool4 = nn.MaxPool2d(kernel_size=[4, 4], stride=4)
        self.pool5 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x1):
        self.in_channels, h, w = x1.size(1), x1.size(2), x1.size(3)

        self.layer1 = self.conv(x1)
        self.layer2 = torch.nn.functional.interpolate(self.conv(self.pool2(x1)), size=(h, w), mode='nearest')
        self.layer3 = torch.nn.functional.interpolate(self.conv(self.pool3(x1)), size=(h, w), mode='nearest')
        self.layer4 = torch.nn.functional.interpolate(self.conv(self.pool4(x1)), size=(h, w), mode='nearest')
        self.layer5 = torch.nn.functional.interpolate(self.conv(self.pool5(x1)), size=(h, w), mode='nearest')

        # self.layer4 = F.interpolate(self.conv(self.pool4(torch.abs(x1-x2))), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, x1], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        nonlinearity = functools.partial(torch.nn.functional.relu, inplace=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CE_Net_res34_PCAM_009(nn.Module):
    def __init__(self, model_path, num_classes=1):
        super(CE_Net_res34_PCAM_009, self).__init__()

        nonlinearity = functools.partial(torch.nn.functional.relu, inplace=True)

        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.sia_sub = Sia_sub_00a(filters[3])

        self.decoder4 = DecoderBlock(5 * (1) + (filters[3]), filters[2])
        self.decoder3 = DecoderBlock(filters[2] * 2, filters[1])
        self.decoder2 = DecoderBlock(filters[1] * 2, filters[0])
        self.decoder1 = DecoderBlock(filters[0] * 2, filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0] * 2, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x_A):
        # Encoder A
        x_A = self.firstconv(x_A)
        x_A = self.firstbn(x_A)
        x_A_c = self.firstrelu(x_A)
        x_A = self.firstmaxpool(x_A_c)
        e1_A = self.encoder1(x_A)
        e2_A = self.encoder2(e1_A)
        e3_A = self.encoder3(e2_A)
        e4_A = self.encoder4(e3_A)

        com_ab = self.sia_sub(e4_A)

        # Decoder
        d4 = self.decoder4(com_ab)
        d3 = self.decoder3(torch.cat([d4, e3_A], 1))
        d2 = self.decoder2(torch.cat([d3, e2_A], 1))
        d1 = self.decoder1(torch.cat([d2, e1_A], 1))

        out = self.finaldeconv1(torch.cat([d1, x_A_c], 1))
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


class BaseNet(nn.Module):
    def __init__(self, resnet_path):
        super(BaseNet, self).__init__()
        self.backbone = CE_Net_res34_PCAM_009(resnet_path)

    def forward(self, x):
        return self.backbone(x)


class Unet(BaseNet):
    def __init__(self, resnet_path):
        super(Unet, self).__init__(resnet_path)
