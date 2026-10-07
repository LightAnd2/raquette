"""Shot types for near-player hits: pose classifier (ServeDetector + RallyClassifier) plus ball geometry.

Both supported camera angles film the near player from behind, so image right is the player's right.
"""
from functools import lru_cache
import logging
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'ml/models/weights'
POSE_MODEL = ROOT / 'ml/train/pose_landmarker_lite.task'
WINDOW_HALF = 8          # frames either side of contact, as in ml/train/extract_poses.py
NEAR_BASELINE = 23.77
SERVICE_LINE = 18.285
LEFT_WRIST, RIGHT_WRIST = 15, 16


@lru_cache(maxsize=1)
def classifiers():
    from ml.models.shot_classifier import RallyClassifier, ServeDetector
    return ServeDetector.load(str(WEIGHTS / 'serve_detector.pt')), RallyClassifier.load(str(WEIGHTS / 'rally_classifier.pt'))


def _pose_estimator():
    import mediapipe as mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.4, min_pose_presence_confidence=.4, min_tracking_confidence=.4)
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _landmarks(estimator, frame, box):
    import mediapipe as mp
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    crop = frame[max(0,int(y1-.1*h)):int(y2+.1*h), max(0,int(x1-.1*w)):int(x2+.1*w)]
    if crop.size == 0:
        return None
    found = estimator.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    if not found.pose_landmarks:
        return None
    return [(l.x, l.y, l.z, l.visibility) for l in found.pose_landmarks[0]]


def pose_windows(video_path, hits, info, size):
    """MediaPipe landmark windows around each contact, cropped to the near player."""
    estimator = _pose_estimator()
    cap = cv2.VideoCapture(str(video_path))
    scale = np.array([info['width']/size[0], info['height']/size[1]] * 2)
    windows = []
    try:
        for hit in hits:
            box = np.array(hit['player_box']) * scale
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, hit['frame'] - WINDOW_HALF))
            window = []
            for _ in range(2*WINDOW_HALF):
                ok, frame = cap.read()
                if not ok:
                    break
                window.append(_landmarks(estimator, frame, box))
            windows.append(window)
    finally:
        cap.release()
        estimator.close()
    return windows


def handedness(hits, windows, kinds):
    """The serving arm is the raised wrist at contact; default right-handed when no serve was seen."""
    votes = []
    for hit, window, kind in zip(hits, windows, kinds):
        contact = window[WINDOW_HALF] if len(window) > WINDOW_HALF else None
        if kind == 'serve' and contact:
            votes.append('left' if contact[LEFT_WRIST][1] < contact[RIGHT_WRIST][1] else 'right')
    return max(set(votes), key=votes.count) if votes else 'right'


def geometry(result, hits):
    """Serve / smash / volley from ball path and court position; side of the body from ball x."""
    track = result['ball_track']
    times = np.array([row['time'] for row in track])
    points = np.array([row['point'] if row['point'] is not None else [np.nan, np.nan] for row in track], float)
    players = [(row['time'], row['players'][0].get('court')) for row in result['player_track'] if row['players']]
    out = []
    for k, hit in enumerate(hits):
        x1, y1, x2, y2 = hit['player_box']
        w, h = x2-x1, y2-y1
        ball_x, ball_y = hit['point']
        above_head = ball_y < y1 + .15*h
        # A toss rises a long way (about a third of the player's height or more) to the highest point of the
        # second before contact, beside the player. A return falls from the far court, so its earliest
        # points are the highest and its only rise is the small bounce.
        window = np.where((times >= hit['time']-1.) & (times < hit['time']) & ~np.isnan(points[:,1]))[0]
        toss = False
        if len(window) >= 5 and np.all(np.abs(points[window,0] - (x1+x2)/2) < 1.5*w):
            apex = int(np.argmin(points[window,1]))
            start = int(np.argmax(points[window[:apex+1],1])) if apex > 0 else 0
            rise = points[window[start:apex+1]]
            steps = np.diff(rise, axis=0)
            # The ball goes almost straight up from the hand: steady climb, little sideways drift.
            toss = (len(rise) >= 4 and rise[0,1] - rise[-1,1] >= .3*h
                    and np.all(steps[:,1] <= 1) and np.all(np.abs(steps[:,0]) <= 4))
        court = min(players, key=lambda p: abs(p[0]-hit['time']))[1] if players else None
        depth = court[1] if court else NEAR_BASELINE
        first = k == 0 or hit['time'] - hits[k-1]['time'] > 4
        if first and above_head and toss:
            kind = 'serve'
        elif above_head and depth < 20:
            kind = 'smash'
        else:
            kind = 'volley' if depth < SERVICE_LINE else 'groundstroke'
        out.append({'kind': kind, 'side': 'right' if ball_x > (x1+x2)/2 else 'left'})
    return out


def shot_name(geometry_info, hand):
    """First release reports serve, forehand and backhand only. Volleys and overheads are named by the
    side of the body they were hit on; the finer geometry kind (volley, smash) stays in `geometry` for later."""
    if geometry_info['kind'] == 'serve':
        return 'serve'
    return 'forehand' if geometry_info['side'] == hand else 'backhand'


def relabel(shots, hand):
    """Apply a user-chosen handedness to already classified shots."""
    return [{**shot, 'type': shot_name(shot['geometry'], hand), 'handedness': hand} for shot in shots]


def classify_shots(video_path, result, hits, info, size, hand=None):
    """hand: 'right' / 'left' from the user; inferred from the serve (else right) when None."""
    if not hits:
        return []
    from ml.models.shot_classifier import landmarks_to_vec
    serve_detector, rally_classifier = classifiers()
    try:
        windows = pose_windows(video_path, hits, info, size)
    except Exception as error:
        # Shot types come from geometry; the pose model only adds recorded probabilities.
        logging.warning('Pose estimation unavailable, continuing without it: %s', error)
        windows = [[] for _ in hits]
    geo = geometry(result, hits)
    hand = hand or handedness(hits, windows, [g['kind'] for g in geo])
    shots = []
    for hit, window, g in zip(hits, windows, geo):
        vectors = [landmarks_to_vec(l) for l in window if l]
        pose = None
        if len(vectors) >= 4:
            proba = rally_classifier.predict_proba(vectors)
            pose = {'serve_probability': round(serve_detector.serve_probability(vectors), 3),
                    'rally': {k: round(float(v), 3) for k, v in proba.items()}}
        shots.append({'hit_id': hit['id'], 'time': hit['time'], 'type': shot_name(g, hand), 'handedness': hand,
                      'geometry': g, 'pose': pose, 'pose_frames': len(vectors)})
    return shots
