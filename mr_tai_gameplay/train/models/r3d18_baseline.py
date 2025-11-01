import torch
import torch.nn as nn
import torchvision
try:
    from mr_tai_gameplay.src.schemas import LABELS
except ImportError:  # when running: python -m train.train_baseline
    from src.schemas import LABELS
    
class R3D18(nn.Module):
    def __init__(self, num_classes: int = len(LABELS), head_only: bool = True):
        super().__init__()
        self.m = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        )
        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, num_classes)

        if head_only:
            for p in self.m.parameters():
                p.requires_grad = False
            # Unfreeze final layer
            for p in self.m.fc.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.m(x)