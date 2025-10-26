import torch
import torch.nn as nn
import torchvision
from src.schemas import LABELS


class R3D18(nn.Module):
    def __init__(self, num_classes: int = len(LABELS)):
        super().__init__()
        self.m = torchvision.models.video.r3d_18(weights=
            torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, num_classes)
    def forward(self, x):
        return self.m(x)