from __future__ import annotations
import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from typing import Dict, List, Tuple
from .schemas import LABELS


class R3D18Baseline(nn.Module):
    """Simple baseline built on torchvision R3D-18 (Kinetics pretrained).
    Last layer replaced with |LABELS| classes.
    """
    def __init__(self, num_classes: int = len(LABELS)):
        super().__init__()
        self.backbone = torchvision.models.video.r3d_18(weights=
        torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        )
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.backbone(x)


class GameplayClassifier:
    def __init__(self, device: str = "cpu", ckpt_path: str | None = None):
        self.device = torch.device(device)
        self.model = R3D18Baseline().to(self.device).eval()
        if ckpt_path:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        # Normalization for Kinetics weights
        w = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        t = w.transforms()
        self._transform = t


    def _sample_clip(self, path: str, t_start: float, t_end: float, num_frames: int = 16) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        s_idx = int(t_start * fps)
        e_idx = max(s_idx+1, int(t_end * fps))
        idxs = np.linspace(s_idx, e_idx-1, num_frames).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError("No frames sampled")
        # To PIL then transform expects (T, H, W, C)
        import PIL.Image
        frames = [PIL.Image.fromarray(f) for f in frames]
        # transforms in weights expect list of PIL Images → torch (C,T,H,W)
        sample = self._transform(frames) # C,T,H,W
        # add batch dim → (B,C,T,H,W)
        return sample.unsqueeze(0)


    def predict(self, video_path: str, t_start: float, t_end: float) -> Dict[str, float]:
        with torch.no_grad():
            clip = self._sample_clip(video_path, t_start, t_end)
            clip = clip.to(self.device)
            logits = self.model(clip)[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
        return {lbl: float(probs[i]) for i, lbl in enumerate(LABELS)}


    @staticmethod
    def topk(probs: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]