"""Video frame extraction and preprocessing utilities."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional


def iter_frames(
    video_path: str,
    skip: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[tuple[int, int]] = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, bgr_array) tuples from a video file.

    Args:
        skip:       yield every Nth frame (1 = every frame)
        max_frames: stop after this many yielded frames
        resize:     (width, height) to resize each frame, or None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    idx = 0
    yielded = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            yield idx, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
        idx += 1

    cap.release()


def video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration_s"] = meta["total_frames"] / meta["fps"] if meta["fps"] else 0
    cap.release()
    return meta


def extract_clip(
    video_path: str,
    start_frame: int,
    end_frame: int,
    out_path: str,
) -> None:
    """Write a sub-clip of a video to disk."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()


def draw_overlay(
    frame: np.ndarray,
    players: list[dict],
    ball: Optional[tuple[float, float]],
    trajectory: list[tuple[float, float]],
    shot_label: Optional[str] = None,
) -> np.ndarray:
    """
    Render detection overlays onto a frame copy.
    Returns a new BGR array — does not modify in place.
    """
    out = frame.copy()

    # Player bounding boxes
    PLAYER_COLORS = [(27, 67, 50), (193, 68, 14)]  # forest green, clay
    for i, player in enumerate(players[:2]):
        bbox = player.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            color = PLAYER_COLORS[i % 2]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                out, f"P{i+1}", (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    # Ball trajectory arc (dashed yellow-green)
    pts = [(int(x), int(y)) for x, y in trajectory if x and y]
    for j in range(1, len(pts)):
        if j % 2 == 0:
            cv2.line(out, pts[j - 1], pts[j], (0, 224, 200), 1, cv2.LINE_AA)

    # Ball position
    if ball:
        bx, by = int(ball[0]), int(ball[1])
        cv2.circle(out, (bx, by), 5, (0, 224, 200), -1, cv2.LINE_AA)
        cv2.circle(out, (bx, by), 6, (27, 67, 50), 1, cv2.LINE_AA)

    # Shot label badge
    if shot_label:
        label = shot_label.upper()
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        pad = 4
        bx = ball[0] if ball else 20
        by_ = (ball[1] - 14) if ball else 20
        x0, y0 = int(bx - tw / 2 - pad), int(by_ - th - pad)
        x1, y1 = int(bx + tw / 2 + pad), int(by_ + pad)
        cv2.rectangle(out, (x0, y0), (x1, y1), (27, 67, 50), -1)
        cv2.putText(
            out, label,
            (int(bx - tw / 2), int(by_)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 247), 1, cv2.LINE_AA,
        )

    return out


def frame_to_jpeg_b64(frame: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR frame as a base64 JPEG string for streaming to the frontend."""
    import base64
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()
