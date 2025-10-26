from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

class EventSegmenter:
    """
    Finds approximate play windows (snap→end) using optical flow energy.
    Video-only: no HUD, no banners.
    """
    def __init__(self,
                flow_sample_stride: int = 2,
                flow_block: int = 15,
                flow_iters: int = 3,
                snap_threshold: float = 1.8, # tune
                end_threshold: float = 0.9, # tune
                min_play_secs: float = 3.0,
                max_play_secs: float = 12.0):
        self.flow_sample_stride = flow_sample_stride
        self.flow_block = flow_block
        self.flow_iters = flow_iters
        self.snap_threshold = snap_threshold
        self.end_threshold = end_threshold
        self.min_play_secs = min_play_secs
        self.max_play_secs = max_play_secs

    @staticmethod
    def _read_frames(path: str, max_fps: float = 30.0):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(fps / max_fps)))
        frames, ts = [], []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            i += 1
        cap.release()
        return np.stack(frames), np.array(ts)
    
    def _flow_energy(self, frames: np.ndarray) -> np.ndarray:
        # Dense Farneback flow
        energies = []
        prev = frames[0]
        for i in range(1, len(frames)):
            if i % self.flow_sample_stride != 0:
                energies.append(energies[-1] if energies else 0.0)
                continue
            h, w = prev.shape
            flow_init = np.zeros((h, w, 2), dtype=np.float32)  # satisfies UMat/ndarray expectation
            flow = cv2.calcOpticalFlowFarneback(
                prev, frames[i],
                flow_init,                 # <-- was None
                pyr_scale=0.5,
                levels=3,
                winsize=int(self.flow_block) | 1,  # ensure odd
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            energies.append(float(np.median(mag)))
            prev = frames[i]
        if not energies:
            energies = [0.0]*(len(frames)-1)
        return np.array([0.0] + energies) # align to frame indices
    
    def segment(self, video_path: str) -> List[Tuple[float, float]]:
        frames, ts = self._read_frames(video_path)
        if len(frames) < 3:
            return []
        e = self._flow_energy(frames)
        # Smooth
        k = 5
        e_s = np.convolve(e, np.ones(k)/k, mode='same')
        # Snap: first strong rise above threshold
        base = np.percentile(e_s, 25)
        high = base * self.snap_threshold
        start_idx = int(np.argmax(e_s > high)) if np.any(e_s > high) else None
        if start_idx is None or start_idx == 0:
            return []
        # End: first sustained drop below end_threshold * base after start
        low = base * self.end_threshold
        end_idx = None
        for j in range(start_idx+5, len(e_s)):
            window = e_s[j:j+5]
            if len(window) < 5:
                break
            if np.all(window < low):
                end_idx = j
                break
        if end_idx is None:
            end_idx = min(len(ts)-1, start_idx + int(self.max_play_secs* (len(ts)/(ts[-1]-ts[0]+1e-6))))
        t_start = ts[start_idx]
        t_end = ts[end_idx]
        if (t_end - t_start) < self.min_play_secs:
            # Try expanding slightly
            t_end = min(ts[-1], t_start + self.min_play_secs)
        return [(t_start, t_end)]