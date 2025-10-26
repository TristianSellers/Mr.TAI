from __future__ import annotations
import json, random
from typing import List, Dict
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision

from src.schemas import LABELS

class ClipDataset(Dataset):
    def __init__(self, index_json: str, num_frames: int = 16):
        self.items: List[Dict] = json.loads(open(index_json).read())
        self.num_frames = num_frames
        w = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        self.tfm = w.transforms()

    def __len__(self):
        return len(self.items)


    def _read_clip(self, path: str, t_start: float, t_end: float):
        import cv2
        import numpy as np
        import torch

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        s = int(t_start * fps)
        e = max(s + 1, int(t_end * fps))
        idxs = np.linspace(s, e - 1, self.num_frames).astype(int)

        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from: {path}")

        # (T, H, W, C) -> Tensor and permute to (T, C, H, W) for video transforms
        vid = torch.from_numpy(np.stack(frames, axis=0))   # uint8, T,H,W,C
        vid = vid.permute(0, 3, 1, 2).contiguous()         # T,C,H,W

        # torchvision video preset returns (C, T, H, W)
        x = self.tfm(vid)
        return x

    def __getitem__(self, i):
        it = self.items[i]
        x = self._read_clip(it["video"], it["t_start"], it["t_end"]) # C,T,H,W
        y = LABELS.index(it["label"])
        return x, y