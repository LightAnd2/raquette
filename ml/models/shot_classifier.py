"""
shot_classifier.py — two-model shot classification system.

Model 1 — ServeDetector:   binary (Serve / Not-Serve)
Model 2 — RallyClassifier: 4-class (Forehand, Backhand, Volley, Smash)

Return is NOT learned visually — it is inferred by the pipeline state machine.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── Class definitions ─────────────────────────────────────────────────────────

RALLY_CLASSES = ["Forehand", "Backhand", "Volley", "Smash"]

# Single-player pose vector: 33 landmarks × 4 (x, y, z, visibility)
LANDMARKS_PER_PLAYER = 33 * 4   # 132
DEFAULT_INPUT_SIZE    = LANDMARKS_PER_PLAYER * 2   # 264 — kept for backward compat


# ── Shared CNN backbone ───────────────────────────────────────────────────────

class _TennisCNN(nn.Module):
    """1D temporal CNN over single-player pose sequences."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(256 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len) for Conv1d
        return self.net(x.permute(0, 2, 1))


# ── Legacy flat architecture (for loading old Kaggle checkpoint) ──────────────

class ShotCNNFlat(nn.Module):
    """Matches the Kaggle-trained 8-class checkpoint (net.* keys)."""

    def __init__(self, input_size: int = DEFAULT_INPUT_SIZE, num_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1))


class ShotCNN(nn.Module):
    """Split conv/head architecture (older local checkpoint format)."""

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
        return self.head(self.conv(x.permute(0, 2, 1)))


# ── Feature extraction ────────────────────────────────────────────────────────

def landmarks_to_vec(landmarks) -> np.ndarray:
    """
    Flatten 33 MediaPipe landmarks → 132-dim float32 vector.
    landmarks: list of (x, y, z, visibility) tuples, length 33
    """
    if not landmarks:
        return np.zeros(LANDMARKS_PER_PLAYER, dtype=np.float32)
    arr = np.array(landmarks, dtype=np.float32).flatten()
    if len(arr) < LANDMARKS_PER_PLAYER:
        arr = np.pad(arr, (0, LANDMARKS_PER_PLAYER - len(arr)))
    return arr[:LANDMARKS_PER_PLAYER]


def extract_dual_player_features(landmarks_p1, landmarks_p2) -> np.ndarray:
    """Concatenate pose vectors for two players → 264-dim vector."""
    return np.concatenate([landmarks_to_vec(landmarks_p1), landmarks_to_vec(landmarks_p2)])


def _prepare_sequence(pose_window: list, input_size: int, target_len: int = 16) -> torch.Tensor:
    """
    Convert a list of pose vectors into a padded (1, target_len, input_size) tensor.
    Each item in pose_window should be a 1-D np.ndarray of length input_size.
    """
    zero = np.zeros(input_size, dtype=np.float32)
    features = []
    for item in pose_window:
        if isinstance(item, np.ndarray) and item.shape[0] == input_size:
            features.append(item.astype(np.float32))
        else:
            # Resize to input_size
            tmp = np.zeros(input_size, dtype=np.float32)
            n = min(len(item), input_size)
            tmp[:n] = np.array(item, dtype=np.float32)[:n]
            features.append(tmp)

    # Pad / truncate to target_len
    while len(features) < target_len:
        features.append(features[-1].copy() if features else zero.copy())
    features = features[:target_len]

    return torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0)


# ── ServeDetector ─────────────────────────────────────────────────────────────

class ServeDetector:
    """
    Binary classifier: Serve vs Not-Serve.
    Trained on single-player 132-dim pose sequences.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.5, device: str = "cpu"):
        self.model     = model.to(device).eval()
        self.threshold = threshold
        self.device    = device

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ServeDetector":
        input_size = LANDMARKS_PER_PLAYER  # 132

        if not Path(path).exists():
            print(f"[ServeDetector] No weights at {path} — using untrained model")
            model = _TennisCNN(input_size, num_classes=2)
            return cls(model, device=device)

        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state      = ckpt['model_state_dict']
            input_size = ckpt.get('input_size', input_size)
            threshold  = ckpt.get('threshold', 0.5)
            model      = _TennisCNN(input_size, num_classes=2)
            model.load_state_dict(state)
            print(f"[ServeDetector] Loaded — input_size={input_size}, threshold={threshold:.2f}")
            return cls(model, threshold=threshold, device=device)

        # Raw state dict fallback
        model = _TennisCNN(input_size, num_classes=2)
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[ServeDetector] Could not load weights: {e}")
        return cls(model, device=device)

    def is_serve(self, pose_window: list) -> bool:
        """Returns True if the pose window looks like a serve."""
        tensor = _prepare_sequence(pose_window, LANDMARKS_PER_PLAYER).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob   = torch.softmax(logits, dim=1)[0, 1].item()   # class 1 = Serve
        return prob >= self.threshold

    def serve_probability(self, pose_window: list) -> float:
        tensor = _prepare_sequence(pose_window, LANDMARKS_PER_PLAYER).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            return torch.softmax(logits, dim=1)[0, 1].item()


# ── RallyClassifier ───────────────────────────────────────────────────────────

class RallyClassifier:
    """
    4-class rally shot classifier: Forehand, Backhand, Volley, Smash.
    Trained on single-player 132-dim pose sequences.
    """

    def __init__(self, model: nn.Module, classes: list, device: str = "cpu"):
        self.model   = model.to(device).eval()
        self.classes = classes
        self.device  = device

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RallyClassifier":
        input_size = LANDMARKS_PER_PLAYER  # 132
        classes    = RALLY_CLASSES

        if not Path(path).exists():
            print(f"[RallyClassifier] No weights at {path} — using untrained model")
            model = _TennisCNN(input_size, num_classes=len(classes))
            return cls(model, classes, device)

        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            classes    = ckpt.get('classes', RALLY_CLASSES)
            input_size = ckpt.get('input_size', input_size)
            model      = _TennisCNN(input_size, num_classes=len(classes))
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"[RallyClassifier] Loaded — {len(classes)} classes: {classes}, input_size={input_size}")
            return cls(model, classes, device)

        model = _TennisCNN(input_size, num_classes=len(classes))
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[RallyClassifier] Could not load weights: {e}")
        return cls(model, classes, device)

    def predict(self, pose_window: list) -> str:
        """Returns the predicted shot type string."""
        tensor = _prepare_sequence(pose_window, LANDMARKS_PER_PLAYER).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            idx    = logits.argmax(dim=1).item()
        return self.classes[idx] if idx < len(self.classes) else "Forehand"

    def predict_proba(self, pose_window: list) -> dict:
        """Returns a dict of {class: probability}."""
        tensor = _prepare_sequence(pose_window, LANDMARKS_PER_PLAYER).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].tolist()
        return {c: round(p, 3) for c, p in zip(self.classes, probs)}


# ── Legacy ShotClassifier (backward compat for old single-model checkpoints) ──

SHOT_CLASSES = ["Forehand", "Backhand", "Serve", "Volley", "Smash", "Slice", "Return", "Tweener"]


class ShotClassifier:
    """
    Legacy single-model classifier kept for backward compat.
    New code should use ServeDetector + RallyClassifier instead.
    """

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

        ckpt = torch.load(path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            classes    = ckpt.get('classes', SHOT_CLASSES)
            input_size = ckpt.get('input_size', DEFAULT_INPUT_SIZE)
            state      = ckpt['model_state_dict']
            if any(k.startswith('net.') for k in state):
                model = ShotCNNFlat(input_size, len(classes))
                print(f"[ShotClassifier] Detected flat (net.*) checkpoint — {len(classes)} classes")
            else:
                model = ShotCNN(input_size, len(classes))
                print(f"[ShotClassifier] Detected split checkpoint — {len(classes)} classes")
            model.load_state_dict(state)
            return cls(model, classes, input_size, device)

        model = ShotCNN(DEFAULT_INPUT_SIZE, len(SHOT_CLASSES))
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[ShotClassifier] Could not load weights: {e}")
        return cls(model, SHOT_CLASSES, DEFAULT_INPUT_SIZE, device)

    def predict(self, pose_sequence: list) -> str:
        tensor = _prepare_sequence(pose_sequence, self.input_size).to(self.device)
        with torch.no_grad():
            idx = self.model(tensor).argmax(dim=1).item()
        return self.classes[idx] if idx < len(self.classes) else "Forehand"
