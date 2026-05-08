"""
pipeline_worker.py — runs the full ML pipeline on a video file.

Called from app.py via asyncio.to_thread() so it doesn't block the event loop.
Heavy imports are deferred so the API starts instantly even without GPU packages.
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

# ── Global model cache (stateless models only — pose_est is per-job) ──────────
_models: dict = {}

def _get_models() -> dict:
    """Load heavy stateless models once and cache them globally."""
    if _models:
        return _models
    print("[pipeline] loading models into cache...")
    from ultralytics import YOLO
    from ml.models.tracknet import BallTracker
    from ml.models.shot_classifier import ShotClassifier

    yolo_weights = str(YOLO_PATH) if YOLO_PATH.exists() else "yolov8n.pt"
    _models["yolo"]       = YOLO(yolo_weights)
    _models["tracker"]    = BallTracker.load(str(TRACKNET_PATH))
    _models["classifier"] = ShotClassifier.load(str(SHOT_CLASSIFIER_PATH))
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
    import mediapipe as mp
    from ml.utils.video import draw_overlay, frame_to_jpeg_b64, video_metadata

    on_progress(0, [], None)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    meta = video_metadata(video_path)
    total = max(meta["total_frames"], 1)
    fps   = meta["fps"] or 30.0

    on_progress(1, [], None)
    models = _get_models()
    player_detector = models["yolo"]
    ball_tracker    = models["tracker"]
    shot_classifier = models["classifier"]
    # MediaPipe Pose is stateful (timestamps) — must be fresh per job
    pose_est = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    on_progress(2, [], None)

    # Reset ball tracker history for this job
    ball_tracker._history.clear()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose_est.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    shots: list[dict] = []
    frame_buffer: list = []
    pose_window: list  = []
    prev_ball: Optional[tuple] = None
    contact_cooldown = 0

    idx  = 0
    SKIP = 3

    try:
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

            frame_h, frame_w = frame.shape[:2]

            # 1 · Player detection
            # Real players: large box (>15% frame height) AND high confidence (>0.55)
            # Ball boys/spectators: smaller, lower confidence, or at edges
            players = []
            yolo_results = player_detector(frame, verbose=False, classes=[0])[0]
            for box in yolo_results.boxes:
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf)
                box_w = bbox[2] - bbox[0]
                box_h = bbox[3] - bbox[1]
                area  = box_w * box_h
                # Must meet both size and confidence thresholds
                if box_h < frame_h * 0.15:
                    continue
                if conf < 0.55:
                    continue
                players.append({"bbox": bbox, "conf": conf, "area": area})
            # Keep 2 largest — players always dominate the frame
            players.sort(key=lambda p: p["area"], reverse=True)
            players = players[:2]

            # 2 · Ball tracking
            ball = None
            trajectory = [s.get("_ball") for s in shots[-10:] if s.get("_ball")]
            if len(frame_buffer) == 3:
                ball = ball_tracker.predict_with_interpolation(frame_buffer)

            # 3 · Pose extraction (both players)
            contact_cooldown = max(0, contact_cooldown - 1)
            lm_p1 = _get_pose(pose_est, frame, players[:1])
            lm_p2 = _get_pose(pose_est, frame, players[1:2])
            if lm_p1 or lm_p2:
                from ml.models.shot_classifier import extract_dual_player_features
                feat = extract_dual_player_features(lm_p1, lm_p2)
                pose_window.append(feat)

            # 4 · Contact detection
            is_contact = _detect_contact(ball, prev_ball, players)
            if not is_contact and contact_cooldown == 0 and players:
                is_contact = _detect_swing(pose_window)

            if is_contact and contact_cooldown == 0 and len(pose_window) >= 4:
                shot_type = shot_classifier.predict(pose_window[-16:])
                speed     = _estimate_speed(ball, prev_ball, fps, SKIP)
                hitter    = _identify_hitter(ball, players)
                cx, cy    = _to_court_coords(ball, frame.shape) if ball else (0.5, 0.5)

                shots.append({
                    "type":    shot_type,
                    "speed":   round(speed),
                    "player":  hitter,
                    "time":    round(idx / fps, 2),
                    "court_x": cx,
                    "court_y": cy,
                    "_ball":   ball,
                })
                pose_window      = pose_window[-4:]
                contact_cooldown = 20

            prev_ball = ball

            # 5 · Stream progress
            pct = max(3, round(idx / total * 100))
            frame_b64 = None
            if idx % (SKIP * 5) == 0:
                annotated = draw_overlay(frame, players, ball, trajectory,
                                         shots[-1]["type"] if shots else None)
                frame_b64 = frame_to_jpeg_b64(annotated, quality=55)

            public_shots = [{k: v for k, v in s.items() if not k.startswith("_")} for s in shots]
            on_progress(pct, public_shots, frame_b64)
            idx += 1

    finally:
        cap.release()
        pose_est.close()

    public_shots = [{k: v for k, v in s.items() if not k.startswith("_")} for s in shots]
    return _build_summary(public_shots)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_swing(pose_window: list) -> bool:
    """Detect a swing from wrist velocity in the pose sequence."""
    if len(pose_window) < 4:
        return False
    try:
        import numpy as np
        recent = pose_window[-4:]
        if isinstance(recent[0], np.ndarray):
            # Dual-player flat vector — wrist indices: left=60, right=64
            lx = [float(v[60]) for v in recent]
            rx = [float(v[64]) for v in recent]
        else:
            lx = [p[15][0] for p in recent]
            rx = [p[16][0] for p in recent]
        l_vel = abs(lx[-1] - lx[0])
        r_vel = abs(rx[-1] - rx[0])
        return max(l_vel, r_vel) > 0.15
    except Exception:
        return False


def _detect_contact(ball, prev_ball, players) -> bool:
    if ball is None or prev_ball is None:
        return False
    dy = prev_ball[1] - ball[1]
    if abs(dy) < 3:
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
    if not players:
        return None
    bbox = players[0]["bbox"]
    h, w = frame.shape[:2]
    # 20% padding around bbox for better pose detection
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    x1 = max(0, int(bbox[0] - bw * 0.1))
    y1 = max(0, int(bbox[1] - bh * 0.1))
    x2 = min(w,  int(bbox[2] + bw * 0.1))
    y2 = min(h,  int(bbox[3] + bh * 0.1))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_est.process(rgb)
    if result.pose_landmarks:
        return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]
    return None


def _estimate_speed(ball, prev_ball, fps: float, skip: int) -> float:
    if ball is None or prev_ball is None:
        return 0.0
    dx = ball[0] - prev_ball[0]
    dy = ball[1] - prev_ball[1]
    pixel_dist = (dx ** 2 + dy ** 2) ** 0.5
    meters_per_pixel = 10.97 / 400   # court width ~10.97m, assumed ~400px in frame
    effective_fps = fps / skip        # one measurement per SKIP frames
    speed_kmh = pixel_dist * meters_per_pixel * effective_fps * 3.6
    return min(speed_kmh, 250.0)     # world record serve is 263 km/h


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
    speeds = [s["speed"] for s in shots if s["speed"] > 0] or [0]
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
    import time
    import random

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
