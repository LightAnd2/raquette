"""
pipeline_worker.py — tennis shot type identifier.

Two-model pipeline:
  1. ServeDetector  — binary (Serve / Not-Serve)
  2. RallyClassifier — 4-class (Forehand, Backhand, Volley, Smash)

RallyStateMachine infers Return from rally context instead of learning it visually.
No ball tracking. Output: [{type, player_name, timestamp}, ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_DIR             = Path(__file__).parent.parent.parent / "ml" / "models" / "weights"
SERVE_DETECTOR_PATH   = MODEL_DIR / "serve_detector.pt"
RALLY_CLASSIFIER_PATH = MODEL_DIR / "rally_classifier.pt"
SHOT_CLASSIFIER_PATH  = MODEL_DIR / "shot_classifier.pt"   # legacy fallback
YOLO_PATH             = MODEL_DIR / "player_detector.pt"

# ── Global model cache (stateless models only) ────────────────────────────────
_models: dict = {}

def _get_models() -> dict:
    if _models:
        return _models
    print("[pipeline] loading models into cache...")
    from ultralytics import YOLO
    from ml.models.shot_classifier import ServeDetector, RallyClassifier, ShotClassifier

    yolo_weights       = str(YOLO_PATH) if YOLO_PATH.exists() else "yolov8n.pt"
    _models["yolo"]    = YOLO(yolo_weights)

    # Prefer new two-model setup; fall back to legacy single model
    if SERVE_DETECTOR_PATH.exists() and RALLY_CLASSIFIER_PATH.exists():
        _models["serve_detector"]   = ServeDetector.load(str(SERVE_DETECTOR_PATH))
        _models["rally_classifier"] = RallyClassifier.load(str(RALLY_CLASSIFIER_PATH))
        _models["mode"]             = "two_model"
        print("[pipeline] using two-model pipeline (ServeDetector + RallyClassifier)")
    else:
        _models["shot_classifier"]  = ShotClassifier.load(str(SHOT_CLASSIFIER_PATH))
        _models["mode"]             = "legacy"
        print("[pipeline] using legacy single-model pipeline")

    print("[pipeline] all models ready")
    return _models


# ── Rally state machine ───────────────────────────────────────────────────────

class RallyStateMachine:
    """
    Infers shot context from rally sequence.

    States:
      SERVE_PENDING   — waiting to see a serve
      RETURN_PENDING  — serve was seen, next shot from other player = Return
      RALLY           — mid-rally, classify normally

    The state resets to SERVE_PENDING after a long gap between shots (> GAP_SEC).
    """

    GAP_SEC = 8.0   # seconds without a shot → assume new point

    def __init__(self):
        self.state       = "serve_pending"
        self.last_slot   = None
        self.last_ts     = None

    def classify(
        self,
        rally_label: str,
        is_serve: bool,
        hitter_slot: int,
        timestamp: float,
    ) -> str:
        """
        Given the rally classifier's prediction and the serve detector's verdict,
        return the final shot label.
        """
        # Reset if big gap between shots (new point)
        if self.last_ts is not None and (timestamp - self.last_ts) > self.GAP_SEC:
            self.state     = "serve_pending"
            self.last_slot = None

        self.last_ts = timestamp

        if self.state == "serve_pending":
            if is_serve:
                self.state     = "return_pending"
                self.last_slot = hitter_slot
                return "Serve"
            else:
                # Mid-rally clip — classify normally, don't transition
                self.last_slot = hitter_slot
                return rally_label

        elif self.state == "return_pending":
            if hitter_slot != self.last_slot:
                # Different player hit — this is the return
                self.state     = "rally"
                self.last_slot = hitter_slot
                return "Return"
            else:
                # Same player hit again (e.g. let, foot-fault replay)
                return "Serve"

        else:  # rally
            self.last_slot = hitter_slot
            return rally_label


# ── Player re-identification tracker ─────────────────────────────────────────

class PlayerTracker:
    """
    Maintains consistent player identities across frames using centroid matching.
    Once a player slot is seeded, only detections close to the last known
    centroid are assigned to that slot — ball boys and spectators that were
    never seeded are ignored.
    """

    MAX_DIST = 250   # pixels

    def __init__(self, player_names: list[str]):
        self.names   = player_names
        self.n       = len(player_names)
        self.slots: list[Optional[dict]] = [None] * self.n

    def assign(self, detections: list[dict]) -> list[Optional[dict]]:
        if not detections:
            return [None] * self.n

        result = [None] * self.n
        used   = set()

        for slot_idx in range(self.n):
            if self.slots[slot_idx] is None:
                continue
            sx = (self.slots[slot_idx]["bbox"][0] + self.slots[slot_idx]["bbox"][2]) / 2
            sy = (self.slots[slot_idx]["bbox"][1] + self.slots[slot_idx]["bbox"][3]) / 2

            best_dist, best_det, best_i = self.MAX_DIST, None, -1
            for i, det in enumerate(detections):
                if i in used:
                    continue
                dx = (det["bbox"][0] + det["bbox"][2]) / 2
                dy = (det["bbox"][1] + det["bbox"][3]) / 2
                dist = ((dx - sx) ** 2 + (dy - sy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_det, best_i = dist, det, i

            if best_det is not None:
                result[slot_idx]     = best_det
                self.slots[slot_idx] = best_det
                used.add(best_i)

        # Lazily seed empty slots with leftover detections
        remaining = [d for i, d in enumerate(detections) if i not in used]
        remaining.sort(key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)
        for slot_idx in range(self.n):
            if self.slots[slot_idx] is None and remaining:
                det = remaining.pop(0)
                self.slots[slot_idx] = det
                result[slot_idx]     = det

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
    from ml.models.shot_classifier import landmarks_to_vec

    n_players    = 4 if mode == "doubles" else 2
    player_names = _resolve_names(player_names, n_players)

    on_progress(0, [], None)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    meta  = video_metadata(video_path)
    total = max(meta["total_frames"], 1)
    fps   = meta["fps"] or 30.0

    on_progress(1, [], None)
    models = _get_models()
    yolo   = models["yolo"]

    use_two_model = models["mode"] == "two_model"
    if use_two_model:
        serve_detector   = models["serve_detector"]
        rally_classifier = models["rally_classifier"]
    else:
        shot_classifier  = models["shot_classifier"]

    # MediaPipe Tasks API (0.10.x+)
    POSE_MODEL = Path(__file__).parent.parent.parent / "ml" / "train" / "pose_landmarker_lite.task"
    BaseOptions        = mp.tasks.BaseOptions
    PoseLandmarker     = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode        = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    pose_est = PoseLandmarker.create_from_options(options)
    on_progress(2, [], None)

    tracker   = PlayerTracker(player_names)
    state_machine = RallyStateMachine()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose_est.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    shots:        list[dict] = []
    pose_windows: list[list] = [[] for _ in range(n_players)]
    cooldowns:    list[int]  = [0] * n_players
    idx  = 0
    SKIP = 2

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx % SKIP != 0:
                idx += 1
                continue

            frame_h, frame_w = frame.shape[:2]

            # 1 · Detect players
            raw_dets = []
            for box in yolo(frame, verbose=False, classes=[0])[0].boxes:
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf)
                bh   = bbox[3] - bbox[1]
                bw   = bbox[2] - bbox[0]
                if bh < frame_h * 0.12 or conf < 0.45:
                    continue
                raw_dets.append({"bbox": bbox, "conf": conf, "area": bw * bh})

            raw_dets.sort(key=lambda d: d["area"], reverse=True)
            raw_dets = raw_dets[:n_players]

            # 2 · Re-ID
            assigned = tracker.assign(raw_dets)

            # 3 · Per-player pose + shot detection
            for slot_idx, player_det in enumerate(assigned):
                cooldowns[slot_idx] = max(0, cooldowns[slot_idx] - 1)

                if player_det is None:
                    continue

                landmarks = _get_pose(pose_est, frame, player_det)
                if landmarks:
                    feat = landmarks_to_vec(landmarks)   # 132-dim single-player vector
                    pose_windows[slot_idx].append(feat)

                if cooldowns[slot_idx] > 0:
                    continue

                if _detect_swing(pose_windows[slot_idx]):
                    window = pose_windows[slot_idx][-16:]
                    if len(window) >= 4:
                        timestamp = round(idx / fps, 2)

                        if use_two_model:
                            is_serve   = serve_detector.is_serve(window)
                            # For state machine, always get rally label too
                            rally_type = rally_classifier.predict(window)
                            shot_type  = state_machine.classify(
                                rally_type, is_serve, slot_idx, timestamp
                            )
                        else:
                            # Legacy single model — use _pad_to_dual
                            padded     = [_pad_to_dual(f) for f in window]
                            shot_type  = shot_classifier.predict(padded)

                        shots.append({
                            "type":        shot_type,
                            "player_name": player_names[slot_idx],
                            "timestamp":   timestamp,
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
    import mediapipe as mp
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
    rgb      = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = pose_est.detect(mp_image)
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        lm = result.pose_landmarks[0]
        return [(l.x, l.y, l.z, l.visibility) for l in lm]
    return None


def _pad_to_dual(single_vec):
    """Pad a 132-dim single-player vector to 264-dim (legacy model compat)."""
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
        # Wrist indices in 132-dim single-player vector: left=60, right=64
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
    SHOT_TYPES   = ["Serve", "Return", "Forehand", "Backhand", "Volley", "Smash"]
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
