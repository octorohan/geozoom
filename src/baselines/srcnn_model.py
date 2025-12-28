import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    Minimal SRCNN-style network
    Input : 4 channels (RGB + NIR)
    Output: 3 channels (RGB)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)
