import random
random.seed(64)

import torch
import torch.nn as nn


class UNet_Light(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(32, 32, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):

        # x: input image

        xe11 = nn.ReLU()(self.e11(x))
        xe12 = nn.ReLU()(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = nn.ReLU()(self.e21(xp1))
        xe22 = nn.ReLU()(self.e22(xe21))
        
        xu1 = self.upconv1(xe22)

        xu11 = torch.cat([xu1, xe12], dim=1)
        xd11 = nn.ReLU()(self.d11(xu11))
        xd12 = nn.ReLU()(self.d12(xd11))

        out = self.outconv(xd12) # generated distortion map

        return out
