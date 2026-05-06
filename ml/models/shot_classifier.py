"""
Temporal shot classifier — identifies shot type from a sliding window of pose sequences.

Architecture: 1D-CNN over joint angle features extracted from MediaPipe landmarks.
Input: sequence of N frames × 33 landmarks × 4 (x, y, z, visibility)
Output: shot type (Forehand, Backhand, Serve, Volley, Smash, Slice)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

SHOT_CLASSES = ["Forehand", "Backhand", "Serve", "Volley", "Smash", "Slice"]

# MediaPipe landmark indices for tennis-relevant joints
JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
}


def extract_joint_angles(landmarks: list) -> np.ndarray:
    """
    Compute joint angles from 33 MediaPipe landmarks.
    Returns a 1D feature vector for one frame.
    """
    if not landmarks:
        return np.zeros(12)

    lm = np.array(landmarks)[:, :3]  # (33, 3) — x, y, z

    def angle(a, b, c):
        v1 = lm[a] - lm[b]
        v2 = lm[c] - lm[b]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.clip(cos, -1, 1))

    return np.array([
        angle(11, 13, 15),  # left elbow
        angle(12, 14, 16),  # right elbow
        angle(13, 11, 23),  # left shoulder-hip
        angle(14, 12, 24),  # right shoulder-hip
        angle(11, 23, 25),  # left hip-knee
        angle(12, 24, 26),  # right hip-knee
        lm[15][0] - lm[16][0],  # wrist horizontal offset
        lm[15][1] - lm[16][1],  # wrist vertical offset
        lm[11][0] - lm[12][0],  # shoulder horizontal span
        lm[23][0] - lm[24][0],  # hip horizontal span
        lm[11][1] - lm[23][1],  # left torso height
        lm[12][1] - lm[24][1],  # right torso height
    ], dtype=np.float32)


class ShotClassifierModel(nn.Module):
    """1D-CNN over temporal pose feature sequences."""

    def __init__(self, input_size: int = 12, seq_len: int = 16, num_classes: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size) → (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        return self.head(self.conv(x))


class ShotClassifier:
    def __init__(self, model: ShotClassifierModel, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ShotClassifier":
        model = ShotClassifierModel()
        if Path(path).exists():
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
        return cls(model, device)

    def predict(self, pose_sequence: list) -> str:
        """
        Args:
            pose_sequence: list of landmark arrays, one per frame
        Returns:
            shot type string
        """
        features = [extract_joint_angles(lm) for lm in pose_sequence]
        # Pad or truncate to fixed length
        target_len = 16
        while len(features) < target_len:
            features.append(features[-1] if features else np.zeros(12))
        features = features[:target_len]

        tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            idx = logits.argmax(dim=1).item()
        return SHOT_CLASSES[idx]
