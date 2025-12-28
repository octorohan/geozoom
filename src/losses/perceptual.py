import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using ResNet18 (frozen).
    Operates on RGB images only.
    """

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Use layers up to layer2 (mid-level features)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

        # Freeze parameters
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.feature_extractor.eval()
        self.criterion = nn.L1Loss()

    def forward(self, fake, real):
        """
        fake, real: (B, 3, H, W) in [0,1]
        """
        f_fake = self.feature_extractor(fake)
        f_real = self.feature_extractor(real)
        return self.criterion(f_fake, f_real)
