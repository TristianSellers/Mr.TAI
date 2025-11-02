from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple


class EventSegmenter:
    """
    Finds approximate play windows (snap→end) using optical flow energy.
    Video-only: no HUD, no banners.

    New:
      • segment_all() returns fixed ~5s windows across the full video for multi-segment parsing.
      • segment_peaks() finds multiple play windows by thresholding smoothed flow energy with
        hysteresis + gap merging (snap→dead-ball), better reflecting real play boundaries.

    segment() keeps your original single-window behavior and falls back to the first fixed window if needed.
    """

    def __init__(
        self,
        flow_sample_stride: int = 2,
        flow_block: int = 15,
        flow_iters: int = 3,
        snap_threshold: float = 1.8,   # used by segment()
        end_threshold: float = 0.9,    # used by segment()
        min_play_secs: float = 3.0,
        max_play_secs: float = 12.0,
    ):
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
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
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
        if not frames:
            return np.empty((0,)), np.array([])
        return np.stack(frames), np.array(ts)

    def _flow_energy(self, frames: np.ndarray) -> np.ndarray:
        # Dense Farneback flow → median magnitude per step (robust to noise)
        energies = []
        if frames.size == 0:
            return np.array([])
        prev = frames[0]
        for i in range(1, len(frames)):
            if i % self.flow_sample_stride != 0:
                energies.append(energies[-1] if energies else 0.0)
                continue
            h, w = prev.shape
            flow_init = np.zeros((h, w, 2), dtype=np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                prev,
                frames[i],
                flow_init,                 # (None upsets some type-checkers)
                pyr_scale=0.5,
                levels=3,
                winsize=int(self.flow_block) | 1,  # ensure odd
                iterations=self.flow_iters,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            energies.append(float(np.median(mag)))
            prev = frames[i]
        if not energies:
            return np.array([])
        # Prepend zero to align energy vector with frame index grid
        return np.array([0.0] + energies)

    # -------------------------------
    # Original single-window behavior
    # -------------------------------
    def segment(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Flow-based single play finder. If no confident window is found,
        falls back to the first fixed 5s window from segment_all().
        """
        try:
            frames, ts = self._read_frames(video_path)
            if len(frames) < 3:
                return self.segment_all(video_path)[:1]
            e = self._flow_energy(frames)
            if e.size == 0:
                return self.segment_all(video_path)[:1]

            # Smooth (simple moving average)
            k = 5
            kernel = np.ones(k) / k
            e_s = np.convolve(e, kernel, mode="same")

            # Baseline + thresholds
            base = np.percentile(e_s, 25)
            high = base * self.snap_threshold

            # Snap: first strong rise above threshold
            start_idx = int(np.argmax(e_s > high)) if np.any(e_s > high) else None
            if start_idx is None or start_idx == 0:
                return self.segment_all(video_path)[:1]

            # End: first sustained drop below end-threshold * base after start
            low = base * self.end_threshold
            end_idx = None
            for j in range(start_idx + 5, len(e_s)):
                window = e_s[j : j + 5]
                if len(window) < 5:
                    break
                if np.all(window < low):
                    end_idx = j
                    break

            if end_idx is None:
                # approximate max window if we never dip low enough
                duration = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
                approx_fps = (len(ts) - 1) / duration if duration > 0 else 30.0
                end_idx = min(
                    len(ts) - 1,
                    start_idx + int(self.max_play_secs * approx_fps),
                )

            t_start = float(ts[start_idx])
            t_end = float(ts[end_idx])
            if (t_end - t_start) < self.min_play_secs:
                t_end = min(float(ts[-1]), t_start + self.min_play_secs)
            return [(t_start, t_end)]
        except Exception:
            # Any failure → fixed-window fallback
            return self.segment_all(video_path)[:1]

    # -------------------------------
    # Fixed windows across the video
    # -------------------------------
    def segment_all(
        self,
        video_path: str,
        window_sec: float = 5.0,
        min_window_sec: float = 2.5
    ) -> List[Tuple[float, float]]:
        """
        Return multiple (t_start, t_end) windows of ~window_sec each across the full video.
        Uses metadata first; falls back to decoding if needed.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()

        if fps > 0 and nframes > 0:
            duration = float(nframes / fps)
        else:
            # Fallback: decode timestamps
            _, ts = self._read_frames(video_path)
            duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

        if duration <= 0.0:
            return []

        windows: List[Tuple[float, float]] = []
        t = 0.0
        while t < duration:
            t_end = min(t + window_sec, duration)
            if (t_end - t) >= min_window_sec:
                windows.append((round(t, 3), round(t_end, 3)))
            t = t_end
        return windows

    # -----------------------------------------
    # New: Play-aware peak/valley segmentation
    # -----------------------------------------
    def segment_peaks(
        self,
        video_path: str,
        smooth_kernel: int = 7,
        enter_mult: float = 1.35,
        exit_mult: float = 0.95,
        min_gap_secs: float = 1.0,
    ) -> List[Tuple[float, float]]:
        """
        Detect multiple play windows from flow energy with hysteresis:

          • Compute dense-flow energy per sampled frame; smooth with moving average.
          • Define baseline = 25th percentile of smoothed energy.
          • Enter a play when energy > enter_mult * baseline.
          • Exit when energy falls below exit_mult * baseline for a few samples.
          • Merge adjacent windows if the gap between them is < min_gap_secs.
          • Enforce min/max play durations via expansion/clamping.

        Returns a list of (t_start, t_end) in seconds.
        """
        frames, ts = self._read_frames(video_path)
        if len(frames) < 3:
            return self.segment_all(video_path)

        e = self._flow_energy(frames)
        if e.size == 0:
            return self.segment_all(video_path)

        # Smooth energy
        k = max(3, smooth_kernel | 1)  # ensure odd and >=3
        kernel = np.ones(k, dtype=np.float32) / float(k)
        e_s = np.convolve(e, kernel, mode="same")

        base = np.percentile(e_s, 25)
        enter_th = base * float(enter_mult)
        exit_th = base * float(exit_mult)

        in_play = False
        start_idx: int | None = None
        windows_idx: List[Tuple[int, int]] = []

        below_count_needed = 3
        below_run = 0

        for i in range(len(e_s)):
            v = e_s[i]
            if not in_play:
                if v > enter_th:
                    in_play = True
                    start_idx = i
                    below_run = 0
            else:
                if v < exit_th:
                    below_run += 1
                else:
                    below_run = 0

                if below_run >= below_count_needed and start_idx is not None:
                    end_idx = max(i - below_count_needed + 1, start_idx + 1)
                    windows_idx.append((start_idx, end_idx))
                    in_play = False
                    start_idx = None
                    below_run = 0

        # If video ended mid-play, close it
        if in_play and start_idx is not None:
            windows_idx.append((start_idx, len(e_s) - 1))

        if not windows_idx:
            return self.segment_all(video_path)

        # Convert idx → timestamps and post-process
        secs_per_step = 0.0
        if len(ts) > 1:
            # average step size in seconds across downsampled timeline
            secs_per_step = float((ts[-1] - ts[0]) / max(1, len(ts) - 1))

        windows: List[Tuple[float, float]] = []
        for s, e_idx in windows_idx:
            t_s = float(ts[min(s, len(ts) - 1)])
            t_e = float(ts[min(e_idx, len(ts) - 1)])
            # Enforce min/max durations
            dur = t_e - t_s
            if dur < self.min_play_secs:
                t_e = min(float(ts[-1]), t_s + self.min_play_secs)
            elif dur > self.max_play_secs:
                t_e = min(float(ts[-1]), t_s + self.max_play_secs)
            windows.append((t_s, t_e))

        # Merge close windows (back-to-back plays or spikes)
        if len(windows) > 1:
            merged: List[Tuple[float, float]] = []
            cur_s, cur_e = windows[0]
            for nxt_s, nxt_e in windows[1:]:
                if (nxt_s - cur_e) <= float(min_gap_secs):
                    # merge
                    cur_e = max(cur_e, nxt_e)
                    # Clamp again to max_play_secs
                    if (cur_e - cur_s) > self.max_play_secs:
                        cur_e = cur_s + self.max_play_secs
                else:
                    merged.append((round(cur_s, 3), round(cur_e, 3)))
                    cur_s, cur_e = nxt_s, nxt_e
            merged.append((round(cur_s, 3), round(cur_e, 3)))
            windows = merged
        else:
            windows = [(round(s, 3), round(e, 3)) for (s, e) in windows]

        return windows
