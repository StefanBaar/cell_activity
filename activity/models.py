import torch
from torch import nn
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract(in_channels, 32, 7, 3)
        self.conv2 = self.contract(32, 64, 3, 1)
        self.conv3 = self.contract(64, 128, 3, 1)

        self.upconv3 = self.expand(128, 64, 3, 1)
        self.upconv2 = self.expand(64*2, 32, 3, 1)
        self.upconv1 = self.expand(32*2, out_channels, 3, 1)

    def __call__(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return contract

    def expand(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                               output_padding=1))
        return expand


class large_UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract(in_channels, 32, 7, 3)
        self.conv2 = self.contract(32 , 64 , 3, 1)
        self.conv3 = self.contract(64 , 128, 3, 1)
        self.conv4 = self.contract(128, 256, 3, 1)
        self.conv5 = self.contract(256, 512, 3, 1)

        self.upconv5 = self.expand(512, 256, 3, 1)
        self.upconv4 = self.expand(512, 128, 3, 1)
        self.upconv3 = self.expand(256,  64, 3, 1)
        self.upconv2 = self.expand(128,  32, 3, 1)
        self.upconv1 = self.expand(64 , out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # upsampling
        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return contract

    def expand(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                               output_padding=1))
        return expand

class super_large_UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract(in_channels, 32, 7, 3)
        self.conv2 = self.contract(32 , 64 , 3, 1)
        self.conv3 = self.contract(64 , 128, 3, 1)
        self.conv4 = self.contract(128, 256, 3, 1)
        self.conv5 = self.contract(256, 512, 3, 1)
        self.conv6 = self.contract(512, 1024, 3, 1)

        self.upconv6 = self.expand(1024, 512, 3, 1)
        self.upconv5 = self.expand(1024, 256, 3, 1)
        self.upconv4 = self.expand(512, 128, 3, 1)
        self.upconv3 = self.expand(256,  64, 3, 1)
        self.upconv2 = self.expand(128,  32, 3, 1)
        self.upconv1 = self.expand(64 , out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        # upsampling
        upconv6 = self.upconv6(conv6)
        upconv5 = self.upconv5(torch.cat([upconv6, conv5], 1))
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return contract

    def expand(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                               output_padding=1))
        return expand
