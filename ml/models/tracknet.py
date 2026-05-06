"""
TrackNet V2 — high-speed small object tracking for tennis ball.

Architecture: VGG16-style encoder + transposed-conv decoder with skip connections.
Input:  3 consecutive RGB frames stacked as a 9-channel tensor (H x W x 9)
Output: single-channel heatmap — Gaussian blob centred on ball, 0 if ball absent

Reference: Huang et al. "TrackNet: A Deep Learning Network for Tracking
High-speed and Tiny Objects in Sports Applications" (2019)
Weights: https://github.com/TrackNetTeam/TrackNet (free, non-commercial)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# ── Architecture ─────────────────────────────────────────────────────────────

def _vgg_block(in_ch: int, out_ch: int, layers: int) -> nn.Sequential:
    mods = []
    for i in range(layers):
        mods += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*mods)


class TrackNetV2(nn.Module):
    """
    Encoder-decoder network for ball heatmap prediction.
    Accepts a (B, 9, H, W) tensor — three stacked RGB frames.
    Returns a (B, 1, H, W) float heatmap in [0, 1].
    """

    def __init__(self):
        super().__init__()

        # Encoder (VGG16-style, adapted for 9-channel input)
        self.enc1 = _vgg_block(9, 64, 2)
        self.enc2 = _vgg_block(64, 128, 2)
        self.enc3 = _vgg_block(128, 256, 3)
        self.enc4 = _vgg_block(256, 512, 3)
        self.enc5 = _vgg_block(512, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder with skip connections
        self.up5 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = _vgg_block(1024, 512, 3)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _vgg_block(768, 256, 3)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _vgg_block(384, 128, 2)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _vgg_block(192, 64, 2)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = _vgg_block(128, 64, 2)

        self.head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Decode with skip connections
        d5 = self.dec5(torch.cat([self.up5(self.pool(e5)), e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


# ── Wrapper ───────────────────────────────────────────────────────────────────

class BallTracker:
    """
    High-level wrapper around TrackNetV2.

    Usage:
        tracker = BallTracker.load("models/tracknet.pt")
        pos = tracker.predict([frame_t_minus_2, frame_t_minus_1, frame_t])
        # pos is (x, y) in pixel coords, or None if ball not visible
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
            state = torch.load(str(p), map_location=device, weights_only=True)
            # Handle both raw state_dict and checkpoint dicts
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
        else:
            print(f"[TrackNet] weights not found at {path} — running untrained (predictions will be noise)")
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
        return pos

    def predict_with_interpolation(self, frames: list) -> Optional[tuple[float, float]]:
        """
        Same as predict() but fills in None positions using linear interpolation
        over the last 5 frames — handles occlusion and motion blur.
        """
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
        stacked = np.concatenate(frames, axis=2)          # (H, W, 9)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)  # (1, 9, H, W)
        return tensor

    def _heatmap_to_coords(
        self, heatmap: np.ndarray, orig_h: int, orig_w: int
    ) -> Optional[tuple[float, float]]:
        if heatmap.max() < self.DETECT_THRESH:
            return None
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Scale back to original resolution
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
