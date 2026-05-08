"""
TrackNet V2 — high-speed small object tracking for tennis ball.

Architecture matches yastrebksv/TrackNet pretrained weights:
  - VGG-style encoder (9-channel input = 3 stacked RGB frames)
  - Bilinear upsampling decoder (no skip connections)
  - 18 conv blocks (conv1..conv18), each with Conv2d + BN + ReLU
  - Output: 256-channel feature map → reduced to 1-channel heatmap at inference

Reference: Huang et al. "TrackNet" (2019)
Weights: https://github.com/yastrebksv/TrackNet
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# ── Architecture ──────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv2d + ReLU + BatchNorm2d — order matches pretrained weight keys
    (block.0=Conv, block.1=ReLU has no params, block.2=BN)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class TrackNetV2(nn.Module):
    """
    Encoder-decoder network matching yastrebksv/TrackNet pretrained weights.
    Input:  (B, 9, H, W) — three RGB frames stacked along channel dim
    Output: (B, 1, H, W) heatmap in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1  = _ConvBlock(9,   64)
        self.conv2  = _ConvBlock(64,  64)
        self.conv3  = _ConvBlock(64,  128)
        self.conv4  = _ConvBlock(128, 128)
        self.conv5  = _ConvBlock(128, 256)
        self.conv6  = _ConvBlock(256, 256)
        self.conv7  = _ConvBlock(256, 256)
        self.conv8  = _ConvBlock(256, 512)
        self.conv9  = _ConvBlock(512, 512)
        self.conv10 = _ConvBlock(512, 512)

        # Decoder
        self.conv11 = _ConvBlock(512, 256)
        self.conv12 = _ConvBlock(256, 256)
        self.conv13 = _ConvBlock(256, 256)
        self.conv14 = _ConvBlock(256, 128)
        self.conv15 = _ConvBlock(128, 128)
        self.conv16 = _ConvBlock(128, 64)
        self.conv17 = _ConvBlock(64,  64)
        self.conv18 = _ConvBlock(64,  256)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.conv2(self.conv1(x))          # 64 ch
        x = self.pool(x)
        x = self.conv4(self.conv3(x))          # 128 ch
        x = self.pool(x)
        x = self.conv7(self.conv6(self.conv5(x)))  # 256 ch
        x = self.pool(x)
        x = self.conv10(self.conv9(self.conv8(x))) # 512 ch

        # Decoder (bilinear upsampling — no skip connections)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv13(self.conv12(self.conv11(x)))   # 256 ch

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv15(self.conv14(x))                # 128 ch

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv17(self.conv16(x))                # 64 ch

        x = self.conv18(x)                             # 256 ch

        # Collapse 256 channels → 1-channel heatmap, min-max normalise to [0,1]
        heatmap = x.mean(dim=1, keepdim=True)
        h_min = heatmap.flatten(1).min(dim=1).values[:, None, None, None]
        h_max = heatmap.flatten(1).max(dim=1).values[:, None, None, None]
        heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
        return heatmap


# ── Wrapper ───────────────────────────────────────────────────────────────────

class BallTracker:
    """
    High-level wrapper around TrackNetV2.

    Usage:
        tracker = BallTracker.load("ml/models/weights/tracknet.pt")
        pos = tracker.predict([frame_t_minus_2, frame_t_minus_1, frame_t])
        # pos → (x, y) in pixel coords, or None if ball not visible
    """

    INPUT_H = 288
    INPUT_W = 512
    DETECT_THRESH = 0.5

    def __init__(self, model: TrackNetV2, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self._history: list[Optional[tuple]] = []

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BallTracker":
        model = TrackNetV2()
        p = Path(path)
        if p.exists():
            state = torch.load(str(p), map_location=device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                model.load_state_dict(state, strict=False)
                print(f"[TrackNet] weights loaded from {path}")
            except Exception as e:
                print(f"[TrackNet] could not load weights: {e} — running untrained")
        else:
            print(f"[TrackNet] weights not found at {path} — running untrained")
        return cls(model, device)

    def predict(self, frames: list) -> Optional[tuple[float, float]]:
        """
        Args:
            frames: list of 3 BGR numpy arrays (H, W, 3), newest last
        Returns:
            (x, y) ball centre in original pixel coords, or None
        """
        if len(frames) < 3:
            return None

        orig_h, orig_w = frames[-1].shape[:2]
        tensor = self._preprocess(frames[-3], frames[-2], frames[-1])

        with torch.no_grad():
            heatmap = self.model(tensor.to(self.device))  # (1, 1, H, W)

        heatmap_np = heatmap.squeeze().cpu().numpy()
        pos = self._heatmap_to_coords(heatmap_np, orig_h, orig_w)
        self._history.append(pos)
        if len(self._history) > 30:   # keep only recent history, prevent memory leak
            self._history = self._history[-30:]
        return pos

    def predict_with_interpolation(self, frames: list) -> Optional[tuple[float, float]]:
        pos = self.predict(frames)
        if pos is not None:
            return pos
        return self._interpolate()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _preprocess(self, f1, f2, f3) -> torch.Tensor:
        import cv2
        frames = []
        for f in (f1, f2, f3):
            resized = cv2.resize(f, (self.INPUT_W, self.INPUT_H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(rgb)
        stacked = np.concatenate(frames, axis=2)           # (H, W, 9)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)  # (1, 9, H, W)
        return tensor

    def _heatmap_to_coords(
        self, heatmap: np.ndarray, orig_h: int, orig_w: int
    ) -> Optional[tuple[float, float]]:
        if heatmap.max() < self.DETECT_THRESH:
            return None
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x = float(x_idx) / self.INPUT_W * orig_w
        y = float(y_idx) / self.INPUT_H * orig_h
        return (x, y)

    def _interpolate(self) -> Optional[tuple[float, float]]:
        recent = self._history[-6:]
        known = [(i, p) for i, p in enumerate(recent) if p is not None]
        if len(known) < 2:
            return None
        i1, p1 = known[-2]
        i2, p2 = known[-1]
        last_none = len(recent) - 1
        t = (last_none - i1) / max(i2 - i1, 1)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        return (x, y)


# ── Training utilities ────────────────────────────────────────────────────────

def make_gaussian_heatmap(
    x: float, y: float, w: int, h: int, sigma: float = 5.0
) -> np.ndarray:
    """Create a 2D Gaussian heatmap for a ball at pixel (x, y)."""
    xs = np.arange(0, w, dtype=np.float32)
    ys = np.arange(0, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    heatmap = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


class TrackNetLoss(nn.Module):
    """Weighted BCE loss — heavily penalises missing the ball."""

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(target)
        weight[target > 0.5] = self.pos_weight
        return F.binary_cross_entropy(pred, target, weight=weight)
