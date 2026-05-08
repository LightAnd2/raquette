"""
Temporal shot classifier — identifies shot type from a sliding window of pose sequences.

Architecture: 1D-CNN over concatenated pose landmarks for up to 2 players.
Input: sequence of N frames × 264 features (2 players × 33 landmarks × 4 values)
Output: shot type (Forehand, Backhand, Serve, Volley, Smash, Slice, Return, Tweener)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

SHOT_CLASSES = ["Forehand", "Backhand", "Serve", "Volley", "Smash", "Slice", "Return", "Tweener"]

# Default landmark feature size per player: 33 landmarks × 4 (x,y,z,visibility)
LANDMARKS_PER_PLAYER = 33 * 4   # 132
DEFAULT_INPUT_SIZE    = LANDMARKS_PER_PLAYER * 2   # 264 — two players


# ── New architecture (matches Colab ShotCNN checkpoint) ───────────────────────

class ShotCNN(nn.Module):
    """1D-CNN over temporal dual-player pose feature sequences."""

    def __init__(self, input_size: int = DEFAULT_INPUT_SIZE, num_classes: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.head(self.conv(x))


class ShotCNNFlat(nn.Module):
    """Same architecture but with a single 'net' Sequential — matches Kaggle-trained checkpoint."""

    def __init__(self, input_size: int = DEFAULT_INPUT_SIZE, num_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)


# ── Legacy architecture (kept for backward compat) ────────────────────────────

class ShotClassifierModel(nn.Module):
    """Original 12-feature joint-angle model."""

    def __init__(self, input_size: int = 12, seq_len: int = 16, num_classes: int = 8):
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
        x = x.permute(0, 2, 1)
        return self.head(self.conv(x))


# ── Feature extraction ────────────────────────────────────────────────────────

def landmarks_to_vec(landmarks) -> np.ndarray:
    """
    Flatten 33 MediaPipe landmarks → 132-dim vector.
    landmarks: list of (x, y, z, visibility) tuples, length 33
    """
    if not landmarks:
        return np.zeros(LANDMARKS_PER_PLAYER, dtype=np.float32)
    arr = np.array(landmarks, dtype=np.float32).flatten()
    # Pad or truncate to exactly 132
    if len(arr) < LANDMARKS_PER_PLAYER:
        arr = np.pad(arr, (0, LANDMARKS_PER_PLAYER - len(arr)))
    return arr[:LANDMARKS_PER_PLAYER]


def extract_dual_player_features(landmarks_p1, landmarks_p2) -> np.ndarray:
    """Concatenate pose vectors for both players → 264-dim vector."""
    v1 = landmarks_to_vec(landmarks_p1)
    v2 = landmarks_to_vec(landmarks_p2)
    return np.concatenate([v1, v2])


# ── Classifier wrapper ────────────────────────────────────────────────────────

class ShotClassifier:
    def __init__(self, model: nn.Module, classes: list, input_size: int, device: str = "cpu"):
        self.model      = model.to(device).eval()
        self.classes    = classes
        self.input_size = input_size
        self.device     = device

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ShotClassifier":
        if not Path(path).exists():
            print(f"[ShotClassifier] No weights at {path} — using untrained model")
            model = ShotCNN(DEFAULT_INPUT_SIZE, len(SHOT_CLASSES))
            return cls(model, SHOT_CLASSES, DEFAULT_INPUT_SIZE, device)

        ckpt = torch.load(path, map_location=device)

        # New checkpoint format: {'model_state_dict': ..., 'classes': ..., 'input_size': ...}
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            classes    = ckpt.get('classes', SHOT_CLASSES)
            input_size = ckpt.get('input_size', DEFAULT_INPUT_SIZE)
            state      = ckpt['model_state_dict']
            # Detect which architecture the checkpoint was trained with
            if any(k.startswith('net.') for k in state):
                model = ShotCNNFlat(input_size, len(classes))
                print(f"[ShotClassifier] Detected flat (net.*) checkpoint")
            else:
                model = ShotCNN(input_size, len(classes))
            model.load_state_dict(state)
            print(f"[ShotClassifier] Loaded checkpoint — {len(classes)} classes, input_size={input_size}")
            return cls(model, classes, input_size, device)

        # Legacy: raw state dict for ShotClassifierModel
        model = ShotClassifierModel(input_size=12, num_classes=len(SHOT_CLASSES))
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[ShotClassifier] Could not load legacy weights: {e}")
        return cls(model, SHOT_CLASSES, 12, device)

    def predict(self, pose_sequence: list) -> str:
        """
        Args:
            pose_sequence: list of feature vectors (one per frame).
                           Each item is either:
                           - a 264-dim np.ndarray (dual-player)
                           - a tuple (landmarks_p1, landmarks_p2) of raw landmark lists
        Returns:
            shot type string
        """
        features = []
        for item in pose_sequence:
            if isinstance(item, np.ndarray):
                feat = item
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                feat = extract_dual_player_features(item[0], item[1])
            else:
                # Single-player fallback — zero-pad second player
                feat = extract_dual_player_features(item, None)
            # Resize to expected input_size
            if len(feat) != self.input_size:
                tmp = np.zeros(self.input_size, dtype=np.float32)
                n = min(len(feat), self.input_size)
                tmp[:n] = feat[:n]
                feat = tmp
            features.append(feat.astype(np.float32))

        # Pad / truncate to 16 frames
        target_len = 16
        while len(features) < target_len:
            features.append(features[-1] if features else np.zeros(self.input_size, dtype=np.float32))
        features = features[:target_len]

        tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            idx    = logits.argmax(dim=1).item()
        return self.classes[idx] if idx < len(self.classes) else "Forehand"
