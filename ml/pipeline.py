"""
Raquette — core inference pipeline
Orchestrates: YOLOv8 (players) → TrackNet (ball) → MediaPipe (pose) → ShotClassifier
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Frame:
    index: int
    image: np.ndarray
    players: list = field(default_factory=list)   # [{bbox, id, team}]
    ball: Optional[tuple] = None                   # (x, y) or None
    trajectory: list = field(default_factory=list) # [(x,y), ...]
    pose_landmarks: Optional[list] = None
    shot: Optional[str] = None
    ball_speed_kmh: Optional[float] = None


@dataclass
class Rally:
    frames: list[Frame] = field(default_factory=list)
    shots: list[dict] = field(default_factory=list)


class RaquettePipeline:
    """
    Full pipeline for tennis video analysis.

    Usage:
        pipeline = RaquettePipeline()
        rally = pipeline.process("match_clip.mp4", on_frame=callback)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._player_detector = None
        self._ball_tracker = None
        self._pose_estimator = None
        self._shot_classifier = None

    def load_models(self):
        """Lazy-load all models to avoid import overhead at startup."""
        self._load_player_detector()
        self._load_ball_tracker()
        self._load_pose_estimator()
        self._load_shot_classifier()

    def _load_player_detector(self):
        from ultralytics import YOLO
        # Fine-tuned on player detection; falls back to base yolov8m
        self._player_detector = YOLO("models/player_detector.pt")

    def _load_ball_tracker(self):
        # TrackNet v2 — specialised for high-speed small object tracking
        # Model weights: https://github.com/TrackNetTeam/TrackNet
        from ml.models.tracknet import TrackNetV2
        self._ball_tracker = TrackNetV2.load("models/tracknet.pt", device=self.device)

    def _load_pose_estimator(self):
        import mediapipe as mp
        self._pose_estimator = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _load_shot_classifier(self):
        from ml.models.shot_classifier import ShotClassifier
        self._shot_classifier = ShotClassifier.load("models/shot_classifier.pt")

    def process(self, video_path: str, on_frame=None, skip_frames: int = 2) -> Rally:
        """
        Process a video and return a Rally with shot-level annotations.

        Args:
            video_path:   Path to the video file.
            on_frame:     Optional callback(frame_index, total, Frame) for live updates.
            skip_frames:  Process every Nth frame for speed (ball tracker interpolates).
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        rally = Rally()
        pose_window: list[Frame] = []  # sliding window for shot classification

        frame_idx = 0
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            frame = Frame(index=frame_idx, image=img)

            # 1. Player detection
            frame.players = self._detect_players(img)

            # 2. Ball tracking
            frame.ball, frame.trajectory = self._track_ball(img, rally.frames)

            # 3. Pose at contact frames
            if frame.ball is not None and self._is_contact_frame(frame, rally.frames):
                frame.pose_landmarks = self._extract_pose(img, frame.players)
                pose_window.append(frame)

                # 4. Shot classification on window
                if len(pose_window) >= 8:
                    shot_type = self._classify_shot(pose_window)
                    frame.shot = shot_type
                    speed = self._estimate_ball_speed(frame, rally.frames, fps)
                    frame.ball_speed_kmh = speed
                    rally.shots.append({
                        "type": shot_type,
                        "speed": round(speed),
                        "frame": frame_idx,
                        "time": round(frame_idx / fps, 2),
                        "player": self._identify_hitter(frame),
                        "court_x": self._to_court_coords(frame.ball, img.shape)[0] if frame.ball else 0.5,
                        "court_y": self._to_court_coords(frame.ball, img.shape)[1] if frame.ball else 0.5,
                    })
                    pose_window = pose_window[-4:]  # keep overlap

            rally.frames.append(frame)

            if on_frame:
                on_frame(frame_idx, total, frame)

            frame_idx += 1

        cap.release()
        return rally

    # ---- Model calls --------------------------------------------------------

    def _detect_players(self, img: np.ndarray) -> list:
        if self._player_detector is None:
            return []
        results = self._player_detector(img, verbose=False)[0]
        players = []
        for box in results.boxes:
            if int(box.cls) == 0:  # class 0 = person
                players.append({"bbox": box.xyxy[0].tolist(), "conf": float(box.conf)})
        return players

    def _track_ball(self, img: np.ndarray, prev_frames: list) -> tuple:
        if self._ball_tracker is None:
            return None, []
        # TrackNet takes 3 consecutive frames as input
        inputs = [f.image for f in prev_frames[-2:]] + [img]
        if len(inputs) < 3:
            return None, []
        ball_pos = self._ball_tracker.predict(inputs)
        trajectory = [f.ball for f in prev_frames[-10:] if f.ball is not None]
        return ball_pos, trajectory

    def _is_contact_frame(self, frame: Frame, prev_frames: list) -> bool:
        if frame.ball is None or not prev_frames:
            return False
        prev = [f.ball for f in prev_frames[-3:] if f.ball is not None]
        if len(prev) < 2:
            return False
        # Detect direction reversal in ball trajectory → contact
        dy_prev = prev[-1][1] - prev[-2][1] if len(prev) >= 2 else 0
        dy_curr = frame.ball[1] - prev[-1][1]
        return dy_prev * dy_curr < 0  # sign change

    def _extract_pose(self, img: np.ndarray, players: list) -> list:
        if self._pose_estimator is None or not players:
            return []
        # Crop to the player closest to the ball
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self._pose_estimator.process(rgb)
        if result.pose_landmarks:
            return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]
        return []

    def _classify_shot(self, window: list[Frame]) -> str:
        if self._shot_classifier is None:
            return "Unknown"
        # Extract joint angles from pose landmarks in window
        pose_seq = [f.pose_landmarks for f in window if f.pose_landmarks]
        return self._shot_classifier.predict(pose_seq)

    def _estimate_ball_speed(self, frame: Frame, prev_frames: list, fps: float) -> float:
        prev = [f for f in prev_frames[-3:] if f.ball is not None]
        if not prev or frame.ball is None:
            return 0.0
        dx = frame.ball[0] - prev[-1].ball[0]
        dy = frame.ball[1] - prev[-1].ball[1]
        pixel_dist = (dx**2 + dy**2) ** 0.5
        # Rough calibration: court width ≈ 10.97m mapped to ~400px
        meters_per_pixel = 10.97 / 400
        meters_per_second = pixel_dist * meters_per_pixel * fps
        return meters_per_second * 3.6  # m/s → km/h

    def _identify_hitter(self, frame: Frame) -> str:
        if not frame.players or frame.ball is None:
            return "P1"
        bx, by = frame.ball
        dists = []
        for i, p in enumerate(frame.players[:2]):
            bbox = p["bbox"]
            px = (bbox[0] + bbox[2]) / 2
            py = (bbox[1] + bbox[3]) / 2
            dists.append(((px - bx) ** 2 + (py - by) ** 2) ** 0.5)
        return "P1" if dists[0] < dists[1] else "P2"

    def _to_court_coords(self, ball_px: tuple, img_shape: tuple) -> tuple:
        h, w = img_shape[:2]
        return round(ball_px[0] / w, 3), round(ball_px[1] / h, 3)
