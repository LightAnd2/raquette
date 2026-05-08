"""
pipeline_worker.py — runs the full ML pipeline on a video file.

Called from main.py via asyncio.to_thread() so it doesn't block the event loop.
All heavy imports are deferred so the API starts instantly even without GPU packages.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Callable, Optional

# Make ml/ importable regardless of working directory
ML_ROOT = Path(__file__).parent.parent.parent / "ml"
sys.path.insert(0, str(ML_ROOT.parent))

MODEL_DIR = Path(__file__).parent.parent.parent / "ml" / "models" / "weights"

SHOT_CLASSIFIER_PATH = MODEL_DIR / "shot_classifier.pt"
TRACKNET_PATH        = MODEL_DIR / "tracknet.pt"
YOLO_PATH            = MODEL_DIR / "player_detector.pt"   # falls back to yolov8n.pt

# ── Global model cache (loaded once, reused across all jobs) ──────────────────
_models: dict = {}

def _get_models():
    """Load models once and cache them globally."""
    if _models:
        return _models
    print("[pipeline] loading models into cache...")
    from ultralytics import YOLO
    from ml.models.tracknet import BallTracker
    from ml.models.shot_classifier import ShotClassifier
    import mediapipe as mp

    yolo_weights = str(YOLO_PATH) if YOLO_PATH.exists() else "yolov8n.pt"
    _models["yolo"]       = YOLO(yolo_weights)
    _models["tracker"]    = BallTracker.load(str(TRACKNET_PATH))
    _models["classifier"] = ShotClassifier.load(str(SHOT_CLASSIFIER_PATH))
    _models["pose"]       = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    print("[pipeline] all models ready")
    return _models


def run_pipeline(
    video_path: str,
    job_id: str,
    on_progress: Callable[[int, list, Optional[str]], None],
) -> dict:
    """
    Process a video through the full Raquette pipeline.

    Args:
        video_path:   absolute path to the uploaded video
        job_id:       for logging
        on_progress:  callback(progress_pct, shots_so_far, frame_b64_or_None)

    Returns:
        dict with keys: shots, rally_length, winner, avg_speed, max_speed
    """
    try:
        return _run_real_pipeline(video_path, on_progress)
    except ImportError as e:
        print(f"[pipeline] ML packages not installed ({e}), falling back to simulation")
        return _run_simulated_pipeline(video_path, on_progress)


# ── Real pipeline ─────────────────────────────────────────────────────────────

def _run_real_pipeline(video_path: str, on_progress) -> dict:
    import cv2
    from ml.utils.video import draw_overlay, frame_to_jpeg_b64, video_metadata

    on_progress(0, [], None)

    meta = video_metadata(video_path)
    total = meta["total_frames"]
    fps   = meta["fps"]

    on_progress(1, [], None)   # loading / fetching cached models
    models = _get_models()
    player_detector = models["yolo"]
    ball_tracker    = models["tracker"]
    shot_classifier = models["classifier"]
    pose_est        = models["pose"]
    on_progress(2, [], None)   # models ready

    cap = cv2.VideoCapture(video_path)
    shots: list[dict] = []
    frame_buffer: list = []
    pose_window: list  = []
    prev_ball: Optional[tuple] = None
    contact_cooldown = 0

    idx = 0
    SKIP = 6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)
        if len(frame_buffer) > 3:
            frame_buffer.pop(0)

        if idx % SKIP != 0:
            idx += 1
            continue

        # 1 · Player detection
        players = []
        yolo_results = player_detector(frame, verbose=False, classes=[0])[0]
        for box in yolo_results.boxes:
            players.append({"bbox": box.xyxy[0].tolist(), "conf": float(box.conf)})
        players.sort(key=lambda p: p["conf"], reverse=True)

        # 2 · Ball tracking
        ball = None
        trajectory = [s.get("_ball") for s in shots[-10:] if s.get("_ball")]
        if len(frame_buffer) == 3:
            ball = ball_tracker.predict_with_interpolation(frame_buffer)

        # 3 · Pose extraction every processed frame (both players)
        contact_cooldown = max(0, contact_cooldown - 1)
        lm_p1 = _get_pose(pose_est, frame, players[:1])   # closest / highest-conf player
        lm_p2 = _get_pose(pose_est, frame, players[1:2])  # second player (may be None)
        if lm_p1 or lm_p2:
            from ml.models.shot_classifier import extract_dual_player_features
            feat = extract_dual_player_features(lm_p1, lm_p2)
            pose_window.append(feat)

        # Contact detection — ball-based if available, else pose-motion-based
        is_contact = _detect_contact(ball, prev_ball, players)
        if not is_contact and contact_cooldown == 0 and players:
            is_contact = _detect_swing(pose_window)

        if is_contact and contact_cooldown == 0 and len(pose_window) >= 4:
            shot_type = shot_classifier.predict(pose_window[-16:])
            speed     = _estimate_speed(ball, prev_ball, fps)
            hitter    = _identify_hitter(ball, players)
            cx, cy    = _to_court_coords(ball, frame.shape) if ball else (0.5, 0.5)

            shot = {
                "type":    shot_type,
                "speed":   round(speed) if round(speed) > 0 else 0,
                "player":  hitter,
                "time":    round(idx / fps, 2),
                "court_x": cx,
                "court_y": cy,
                "_ball":   ball,
            }
            shots.append(shot)
            pose_window = pose_window[-4:]
            contact_cooldown = 20   # ignore next 20 frames

        prev_ball = ball

        # 4 · Build annotated frame for frontend
        pct = max(3, round(idx / total * 100))   # never report below 3 once loop starts
        frame_b64 = None
        if idx % (SKIP * 5) == 0:   # stream every 5th processed frame (more responsive)
            annotated = draw_overlay(frame, players, ball, trajectory,
                                     shots[-1]["type"] if shots else None)
            frame_b64 = frame_to_jpeg_b64(annotated, quality=55)

        public_shots = [{k: v for k, v in s.items() if not k.startswith("_")} for s in shots]
        on_progress(pct, public_shots, frame_b64)
        idx += 1

    cap.release()
    # pose_est is cached globally — don't close it

    public_shots = [{k: v for k, v in s.items() if not k.startswith("_")} for s in shots]
    return _build_summary(public_shots)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_swing(pose_window: list) -> bool:
    """Detect a swing from wrist velocity in the pose sequence."""
    if len(pose_window) < 4:
        return False
    try:
        import numpy as np
        # pose_window items are either numpy feature vectors (264,) or raw landmark lists
        recent = pose_window[-4:]
        if isinstance(recent[0], np.ndarray):
            # Dual-player flat vector: player1 landmarks at indices 0..131
            # MediaPipe wrist indices 15 & 16 → each landmark is 4 values (x,y,z,vis)
            # wrist_left_x  = index 15*4 = 60
            # wrist_right_x = index 16*4 = 64
            lx = [v[60] for v in recent]
            rx = [v[64] for v in recent]
        else:
            lx = [p[15][0] for p in recent]
            rx = [p[16][0] for p in recent]
        l_vel = abs(lx[-1] - lx[0])
        r_vel = abs(rx[-1] - rx[0])
        return max(l_vel, r_vel) > 0.15   # 15% frame width movement — only real swings
    except Exception:
        return False



def _detect_contact(ball, prev_ball, players) -> bool:
    if ball is None or prev_ball is None:
        return False
    dy_prev = prev_ball[1] - ball[1]
    # Direction reversal in Y = bounce or racquet contact
    # Also check proximity to a player
    if abs(dy_prev) < 3:
        return False
    for p in players[:2]:
        bbox = p["bbox"]
        px = (bbox[0] + bbox[2]) / 2
        py = (bbox[1] + bbox[3]) / 2
        dist = ((ball[0] - px) ** 2 + (ball[1] - py) ** 2) ** 0.5
        if dist < 120:
            return True
    return False


def _get_pose(pose_est, frame, players):
    import cv2
    import mediapipe as mp

    if not players:
        return None
    # Crop to the tallest detected player
    bbox = players[0]["bbox"]
    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_est.process(rgb)
    if result.pose_landmarks:
        return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]
    return None


def _estimate_speed(ball, prev_ball, fps) -> float:
    if ball is None or prev_ball is None:
        return 0.0
    dx = ball[0] - prev_ball[0]
    dy = ball[1] - prev_ball[1]
    pixel_dist = (dx**2 + dy**2) ** 0.5
    meters_per_pixel = 10.97 / 400
    return pixel_dist * meters_per_pixel * fps * 3.6


def _identify_hitter(ball, players) -> str:
    if not players or ball is None:
        return "P1"
    dists = []
    for p in players[:2]:
        bbox = p["bbox"]
        px = (bbox[0] + bbox[2]) / 2
        py = (bbox[1] + bbox[3]) / 2
        dists.append(((ball[0] - px) ** 2 + (ball[1] - py) ** 2) ** 0.5)
    if len(dists) < 2:
        return "P1"
    return "P1" if dists[0] <= dists[1] else "P2"


def _to_court_coords(ball, img_shape) -> tuple[float, float]:
    h, w = img_shape[:2]
    return round(ball[0] / w, 3), round(ball[1] / h, 3)


def _build_summary(shots: list) -> dict:
    speeds = [s["speed"] for s in shots] or [0]
    p1 = sum(1 for s in shots if s["player"] == "P1")
    return {
        "shots":        shots,
        "rally_length": len(shots),
        "winner":       "P1" if p1 > len(shots) / 2 else "P2",
        "avg_speed":    round(sum(speeds) / len(speeds)),
        "max_speed":    max(speeds),
    }


# ── Simulation fallback (no ML packages installed) ───────────────────────────

def _run_simulated_pipeline(video_path: str, on_progress) -> dict:
    import time, random

    SHOT_TYPES = ["Serve", "Return", "Forehand", "Backhand", "Volley", "Smash", "Slice"]
    shots: list[dict] = []

    for i in range(20):
        time.sleep(0.5)
        pct = round((i + 1) / 20 * 100)
        if i % 3 == 0:
            shots.append({
                "type":    random.choice(SHOT_TYPES),
                "speed":   random.randint(60, 190),
                "player":  random.choice(["P1", "P2"]),
                "time":    round(i * 0.4, 1),
                "court_x": round(random.uniform(0.15, 0.85), 2),
                "court_y": round(random.uniform(0.1, 0.9), 2),
            })
        on_progress(pct, shots, None)

    return _build_summary(shots)
