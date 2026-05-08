"""
pipeline_worker.py — tennis shot type identifier.

Detects players via YOLO, tracks them consistently across frames (re-ID),
extracts pose with MediaPipe, and classifies each shot with the temporal CNN.
No ball tracking. Output: [{type, player_name, timestamp}, ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_DIR            = Path(__file__).parent.parent.parent / "ml" / "models" / "weights"
SHOT_CLASSIFIER_PATH = MODEL_DIR / "shot_classifier.pt"
YOLO_PATH            = MODEL_DIR / "player_detector.pt"

# ── Global model cache (stateless models only) ────────────────────────────────
_models: dict = {}

def _get_models() -> dict:
    if _models:
        return _models
    print("[pipeline] loading models into cache...")
    from ultralytics import YOLO
    from ml.models.shot_classifier import ShotClassifier

    yolo_weights     = str(YOLO_PATH) if YOLO_PATH.exists() else "yolov8n.pt"
    _models["yolo"]       = YOLO(yolo_weights)
    _models["classifier"] = ShotClassifier.load(str(SHOT_CLASSIFIER_PATH))
    print("[pipeline] all models ready")
    return _models


# ── Player re-identification tracker ─────────────────────────────────────────

class PlayerTracker:
    """
    Maintains consistent player identities across frames using centroid matching.
    Once a player slot is seeded, only detections close to the last known
    centroid are assigned to that slot — ball boys and spectators that were
    never seeded are ignored.
    """

    MAX_DIST = 250   # pixels — max centroid movement between frames

    def __init__(self, player_names: list[str]):
        self.names   = player_names
        self.n       = len(player_names)
        self.slots: list[Optional[dict]] = [None] * self.n   # last known bbox per slot
        self._seeded = False

    def assign(self, detections: list[dict]) -> list[Optional[dict]]:
        """
        Match detections to player slots.
        Returns list of length self.n — None for slots not visible this frame.
        """
        if not detections:
            return [None] * self.n

        # First frame: seed slots by left-to-right x position
        if not self._seeded:
            valid = [d for d in detections if d is not None][:self.n]
            valid.sort(key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)
            for i, det in enumerate(valid):
                self.slots[i] = det
            self._seeded = len(valid) > 0
            return list(self.slots)

        result = [None] * self.n
        used   = set()

        for slot_idx in range(self.n):
            if self.slots[slot_idx] is None:
                continue
            sx = (self.slots[slot_idx]["bbox"][0] + self.slots[slot_idx]["bbox"][2]) / 2
            sy = (self.slots[slot_idx]["bbox"][1] + self.slots[slot_idx]["bbox"][3]) / 2

            best_dist = self.MAX_DIST
            best_det  = None
            best_i    = -1

            for i, det in enumerate(detections):
                if i in used:
                    continue
                dx = (det["bbox"][0] + det["bbox"][2]) / 2
                dy = (det["bbox"][1] + det["bbox"][3]) / 2
                dist = ((dx - sx) ** 2 + (dy - sy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_det  = det
                    best_i    = i

            if best_det is not None:
                result[slot_idx]        = best_det
                self.slots[slot_idx]    = best_det
                used.add(best_i)

        return result


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    job_id: str,
    on_progress: Callable[[int, list, Optional[str]], None],
    mode: str = "singles",
    player_names: Optional[list[str]] = None,
) -> dict:
    try:
        return _run_real_pipeline(video_path, on_progress, mode, player_names)
    except ImportError as e:
        print(f"[pipeline] ML packages not installed ({e}), falling back to simulation")
        return _run_simulated_pipeline(video_path, on_progress, mode, player_names)


# ── Real pipeline ─────────────────────────────────────────────────────────────

def _run_real_pipeline(
    video_path: str,
    on_progress,
    mode: str,
    player_names: Optional[list[str]],
) -> dict:
    import cv2
    import mediapipe as mp
    from ml.utils.video import video_metadata

    n_players     = 4 if mode == "doubles" else 2
    player_names  = _resolve_names(player_names, n_players)

    on_progress(0, [], None)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    meta  = video_metadata(video_path)
    total = max(meta["total_frames"], 1)
    fps   = meta["fps"] or 30.0

    on_progress(1, [], None)
    models          = _get_models()
    player_detector = models["yolo"]
    shot_classifier = models["classifier"]

    # MediaPipe Pose is stateful — must be fresh per job
    pose_est = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    on_progress(2, [], None)

    tracker = PlayerTracker(player_names)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose_est.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    shots:        list[dict] = []
    pose_windows: list[list] = [[] for _ in range(n_players)]
    cooldowns:    list[int]  = [0] * n_players
    idx  = 0
    SKIP = 3

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx % SKIP != 0:
                idx += 1
                continue

            frame_h, frame_w = frame.shape[:2]

            # 1 · Detect and filter players
            raw_dets = []
            yolo_results = player_detector(frame, verbose=False, classes=[0])[0]
            for box in yolo_results.boxes:
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf)
                bh   = bbox[3] - bbox[1]
                bw   = bbox[2] - bbox[0]
                if bh < frame_h * 0.12 or conf < 0.45:
                    continue
                raw_dets.append({"bbox": bbox, "conf": conf, "area": bw * bh})

            # Sort by area — largest boxes are the actual players
            raw_dets.sort(key=lambda d: d["area"], reverse=True)
            raw_dets = raw_dets[:n_players]

            # 2 · Re-ID: assign detections to consistent player slots
            assigned = tracker.assign(raw_dets)

            # 3 · Per-player pose + swing detection
            for slot_idx, player_det in enumerate(assigned):
                cooldowns[slot_idx] = max(0, cooldowns[slot_idx] - 1)

                if player_det is None:
                    continue

                landmarks = _get_pose(pose_est, frame, player_det)
                if landmarks:
                    from ml.models.shot_classifier import landmarks_to_vec
                    feat = landmarks_to_vec(landmarks)
                    pose_windows[slot_idx].append(feat)

                if cooldowns[slot_idx] > 0:
                    continue

                if _detect_swing(pose_windows[slot_idx]):
                    window = pose_windows[slot_idx][-16:]
                    if len(window) >= 4:
                        shot_type = shot_classifier.predict(
                            [_pad_to_dual(f) for f in window]
                        )
                        shots.append({
                            "type":        shot_type,
                            "player_name": player_names[slot_idx],
                            "timestamp":   round(idx / fps, 2),
                        })
                        pose_windows[slot_idx] = pose_windows[slot_idx][-4:]
                        cooldowns[slot_idx]    = 20

            pct = max(3, round(idx / total * 100))
            on_progress(pct, list(shots), None)
            idx += 1

    finally:
        cap.release()
        pose_est.close()

    return _build_summary(shots)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_names(names: Optional[list[str]], n: int) -> list[str]:
    defaults = [f"P{i+1}" for i in range(n)]
    if not names:
        return defaults
    result = []
    for i in range(n):
        val = names[i].strip() if i < len(names) and names[i].strip() else defaults[i]
        result.append(val)
    return result


def _get_pose(pose_est, frame, player_det):
    import cv2
    bbox = player_det["bbox"]
    h, w = frame.shape[:2]
    bw   = bbox[2] - bbox[0]
    bh   = bbox[3] - bbox[1]
    x1   = max(0, int(bbox[0] - bw * 0.1))
    y1   = max(0, int(bbox[1] - bh * 0.1))
    x2   = min(w,  int(bbox[2] + bw * 0.1))
    y2   = min(h,  int(bbox[3] + bh * 0.1))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_est.process(rgb)
    if result.pose_landmarks:
        return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]
    return None


def _pad_to_dual(single_vec):
    """Pad a 132-dim single-player vector to 264-dim dual-player vector."""
    import numpy as np
    from ml.models.shot_classifier import DEFAULT_INPUT_SIZE, LANDMARKS_PER_PLAYER
    out = np.zeros(DEFAULT_INPUT_SIZE, dtype=np.float32)
    n   = min(len(single_vec), LANDMARKS_PER_PLAYER)
    out[:n] = single_vec[:n]
    return out


def _detect_swing(pose_window: list) -> bool:
    if len(pose_window) < 4:
        return False
    try:
        import numpy as np
        recent = pose_window[-4:]
        # Single-player 132-dim vector: wrist_left=index 60, wrist_right=index 64
        lx = [float(v[60]) for v in recent]
        rx = [float(v[64]) for v in recent]
        return max(abs(lx[-1] - lx[0]), abs(rx[-1] - rx[0])) > 0.15
    except Exception:
        return False


def _build_summary(shots: list) -> dict:
    return {
        "shots":        shots,
        "rally_length": len(shots),
    }


# ── Simulation fallback ───────────────────────────────────────────────────────

def _run_simulated_pipeline(
    video_path: str,
    on_progress,
    mode: str,
    player_names: Optional[list[str]],
) -> dict:
    import time
    import random

    n_players    = 4 if mode == "doubles" else 2
    player_names = _resolve_names(player_names, n_players)
    SHOT_TYPES   = ["Serve", "Return", "Forehand", "Backhand", "Volley", "Smash", "Slice"]
    shots: list[dict] = []

    for i in range(20):
        time.sleep(0.4)
        pct = round((i + 1) / 20 * 100)
        if i % 3 == 0:
            shots.append({
                "type":        random.choice(SHOT_TYPES),
                "player_name": random.choice(player_names),
                "timestamp":   round(i * 0.5, 1),
            })
        on_progress(pct, list(shots), None)

    return _build_summary(shots)
