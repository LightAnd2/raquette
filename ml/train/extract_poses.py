"""
extract_poses.py — extract pose sequences from labeled tennis clips.

Reads labels.csv, runs YOLO + MediaPipe on a window around each labeled frame,
and saves pose sequences to poses.pkl for training on Kaggle.

Usage:
    python ml/train/extract_poses.py

Output:
    ml/train/poses.pkl  — dict with keys:
        'X'       : np.ndarray (N, 16, 132) — pose sequences
        'y_serve' : np.ndarray (N,)          — binary (1=Serve, 0=Not-Serve)
        'y_rally' : np.ndarray (N,)          — 4-class index (for non-serve shots)
        'labels'  : list[str]                — original label strings
        'classes' : list[str]                — RALLY_CLASSES
"""

import sys
import csv
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Make ml/ importable
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CLIPS_DIR    = Path.home() / "Desktop" / "tennis_clips"
LABELS_CSV   = CLIPS_DIR / "labels.csv"
OUTPUT_PKL   = Path(__file__).parent / "poses.pkl"

WINDOW_HALF  = 8    # frames before and after the labeled frame
TARGET_LEN   = 16   # total sequence length
YOLO_WEIGHTS = str(ROOT / "ml" / "models" / "weights" / "player_detector.pt")

RALLY_CLASSES  = ["Forehand", "Backhand", "Volley", "Smash"]
# Labels that go into the serve detector (treated as "Serve")
SERVE_LABELS   = {"Serve"}
# Labels that go into the rally classifier
RALLY_LABELS   = set(RALLY_CLASSES)
# Labels to include in training (Return/Slice/Tweener used as non-serve negatives only)
INCLUDE_LABELS = SERVE_LABELS | RALLY_LABELS | {"Return", "Slice", "Tweener"}


def load_yolo():
    from ultralytics import YOLO
    if Path(YOLO_WEIGHTS).exists():
        return YOLO(YOLO_WEIGHTS)
    return YOLO("yolov8n.pt")


POSE_MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"

def load_mediapipe():
    """Load MediaPipe Pose using the new Tasks API (mediapipe 0.10.x)."""
    import mediapipe as mp
    import urllib.request

    if not POSE_MODEL_PATH.exists():
        print("  Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, POSE_MODEL_PATH)
        print("  Downloaded.")

    BaseOptions        = mp.tasks.BaseOptions
    PoseLandmarker     = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode        = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return PoseLandmarker.create_from_options(options)


def get_largest_player_bbox(yolo, frame):
    """Return the largest (most likely to be a player) bounding box in the frame."""
    import cv2
    frame_h = frame.shape[0]
    results  = yolo(frame, verbose=False, classes=[0])[0]
    best     = None
    best_area = 0
    for box in results.boxes:
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf)
        bh   = bbox[3] - bbox[1]
        bw   = bbox[2] - bbox[0]
        if bh < frame_h * 0.10 or conf < 0.40:
            continue
        area = bw * bh
        if area > best_area:
            best_area = area
            best = bbox
    return best


def extract_landmarks(pose_est, frame, bbox):
    """Crop to player bbox and run MediaPipe. Returns 132-dim vector or None."""
    import cv2
    import mediapipe as mp
    h, w = frame.shape[:2]
    bw   = bbox[2] - bbox[0]
    bh   = bbox[3] - bbox[1]
    x1   = max(0, int(bbox[0] - bw * 0.1))
    y1   = max(0, int(bbox[1] - bh * 0.1))
    x2   = min(w, int(bbox[2] + bw * 0.1))
    y2   = min(h, int(bbox[3] + bh * 0.1))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb      = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = pose_est.detect(mp_image)
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        lm  = result.pose_landmarks[0]
        arr = np.array([(l.x, l.y, l.z, l.visibility) for l in lm], dtype=np.float32).flatten()
        return arr[:132]
    return None


def extract_sequence(cap, yolo, pose_est, center_frame: int, fps: float):
    """Extract a 16-frame pose sequence centered on center_frame."""
    import cv2
    start = max(0, center_frame - WINDOW_HALF)
    end   = start + TARGET_LEN

    sequence = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    cached_bbox = None   # reuse bbox if YOLO misses on nearby frames

    for f_idx in range(start, end):
        ret, frame = cap.read()
        if not ret:
            # Pad with zeros if video ends early
            sequence.append(np.zeros(132, dtype=np.float32))
            continue

        bbox = get_largest_player_bbox(yolo, frame) or cached_bbox
        if bbox is None:
            sequence.append(np.zeros(132, dtype=np.float32))
            continue

        cached_bbox = bbox
        vec = extract_landmarks(pose_est, frame, bbox)
        sequence.append(vec if vec is not None else np.zeros(132, dtype=np.float32))

    # Ensure exactly TARGET_LEN frames
    while len(sequence) < TARGET_LEN:
        sequence.append(np.zeros(132, dtype=np.float32))
    return np.array(sequence[:TARGET_LEN], dtype=np.float32)   # (16, 132)


def main():
    print(f"Reading labels from {LABELS_CSV}")
    with open(LABELS_CSV, newline='') as f:
        rows = list(csv.DictReader(f))

    # Filter to included labels
    rows = [r for r in rows if r['label'] in INCLUDE_LABELS]
    print(f"  {len(rows)} usable labels (out of {sum(1 for _ in open(LABELS_CSV))-1} total)")

    by_label = Counter(r['label'] for r in rows)
    for label, count in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")

    # Group by video
    by_video = defaultdict(list)
    for r in rows:
        by_video[r['video']].append(r)

    print(f"\nLoading YOLO...")
    yolo = load_yolo()
    print(f"Loading MediaPipe...")
    pose_est = load_mediapipe()

    X       = []
    labels  = []
    y_serve = []
    y_rally = []

    total_rows = len(rows)
    processed  = 0
    skipped    = 0

    for video_name, video_rows in by_video.items():
        video_path = CLIPS_DIR / video_name
        if not video_path.exists():
            print(f"  [skip] {video_name} — file not found")
            skipped += len(video_rows)
            continue

        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"\n  ▶ {video_name} ({len(video_rows)} labels)")

        for row in video_rows:
            center = int(row['frame'])
            label  = row['label']

            seq = extract_sequence(cap, yolo, pose_est, center, fps)
            X.append(seq)
            labels.append(label)

            # Serve detector target
            y_serve.append(1 if label in SERVE_LABELS else 0)

            # Rally classifier target (only meaningful for RALLY_LABELS)
            if label in RALLY_LABELS:
                y_rally.append(RALLY_CLASSES.index(label))
            else:
                y_rally.append(-1)   # -1 = ignore (non-rally label)

            processed += 1
            if processed % 50 == 0:
                print(f"    {processed}/{total_rows} extracted...")

        cap.release()

    pose_est.close()

    X       = np.array(X, dtype=np.float32)        # (N, 16, 132)
    y_serve = np.array(y_serve, dtype=np.int64)
    y_rally = np.array(y_rally, dtype=np.int64)

    data = {
        'X':       X,
        'y_serve': y_serve,
        'y_rally': y_rally,
        'labels':  labels,
        'classes': RALLY_CLASSES,
    }

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✅ Saved {len(X)} sequences to {OUTPUT_PKL}")
    print(f"   Shape: X={X.shape}, y_serve={y_serve.shape}, y_rally={y_rally.shape}")
    print(f"   Serve positives:  {y_serve.sum()} / {len(y_serve)}")
    print(f"   Rally breakdown:  {Counter(labels[i] for i in range(len(labels)) if y_rally[i] >= 0)}")
    print(f"   Skipped: {skipped} (video files not found)")


if __name__ == "__main__":
    main()
