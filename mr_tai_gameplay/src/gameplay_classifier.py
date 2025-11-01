from __future__ import annotations
import torch
import torchvision
import cv2
import numpy as np
from typing import Dict, List, Tuple
from .schemas import LABELS
from mr_tai_gameplay.train.models.r3d18_baseline import R3D18


class GameplayClassifier:
    def __init__(self, device: str = "cpu", ckpt_path: str | None = None):
        self.device = torch.device(device)
        # Use SAME architecture as training so checkpoint keys match
        self.model = R3D18(num_classes=len(LABELS), head_only=False).to(self.device).eval()

        if ckpt_path:
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state, strict=True)

        # Kinetics-400 video preset transforms (expects Tensor video: T,C,H,W)
        w = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        self._transform = w.transforms()

    def _sample_clip(self, path: str, t_start: float, t_end: float, num_frames: int = 32) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        s_idx = int(t_start * fps)
        e_idx = max(s_idx + 1, int(t_end * fps))
        span = e_idx - s_idx
        # Front-bias: emphasize early frames (handoff)
        idxs = np.linspace(s_idx, s_idx + int(0.6 * span) - 1, num_frames).astype(int)

        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames could be read from {path}")

        # (T, H, W, C) -> (T, C, H, W) tensor for video transforms
        vid = torch.from_numpy(np.stack(frames, axis=0))  # uint8
        vid = vid.permute(0, 3, 1, 2).contiguous()        # T, C, H, W

        # Preset returns (C, T, H, W)
        sample = self._transform(vid)                     # C, T, H, W
        return sample.unsqueeze(0)                        # B, C, T, H, W

    def predict(self, video_path: str, t_start: float, t_end: float, num_frames: int = 32) -> Dict[str, float]:
        with torch.no_grad():
            clip = self._sample_clip(video_path, t_start, t_end, num_frames=num_frames)
            clip = clip.to(self.device)
            logits = self.model(clip)[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
        return {lbl: float(probs[i]) for i, lbl in enumerate(LABELS)}

    @staticmethod
    def topk(probs: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]
