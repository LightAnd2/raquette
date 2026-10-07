"""Train the bounce classifier on TrackNet dataset games 1-7 using this pipeline's own ball tracks.

  python scripts/train_bounce.py extract   # TrackNet ball tracks for every training clip (cached)
  python scripts/train_bounce.py train     # fit CatBoost, save ml/models/weights/bounce_classifier.cbm

Games 8-10 are never touched here; scripts/eval_tracknet.py scores on them.
"""
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_tracknet import DATA, labels  # noqa: E402

TRAIN_GAMES = [f'game{i}' for i in range(1, 8)]
TRACKS = ROOT / 'datasets/tracknet/train_tracks'
MODEL = ROOT / 'ml/models/weights/bounce_classifier.cbm'


def train_clips():
    for game in TRAIN_GAMES:
        for clip in sorted((DATA / game).iterdir(), key=lambda p: int(p.name.replace('Clip', ''))):
            yield game, clip


def extract():
    import torch
    from backend.app import analytics as A
    ball_model = A.models()[0]
    TRACKS.mkdir(parents=True, exist_ok=True)
    for game, clip in train_clips():
        out = TRACKS / f'{game}_{clip.name}.json'
        if out.exists():
            continue
        frames, track = deque(maxlen=3), []
        with torch.inference_mode():
            for index, path in enumerate(sorted(clip.glob('*.jpg'))):
                frames.append(cv2.resize(cv2.imread(str(path)), (640, 360)))
                point = None
                if len(frames) == 3:
                    labels_map = ball_model(A.tensor(np.concatenate(list(frames)[::-1], axis=2)).to(next(ball_model.parameters()).dtype))[0].argmax(0).cpu().numpy().astype(np.uint8)
                    point = A.choose_ball(A.ball_candidates(labels_map), track, index / 30)
                track.append({'time': index / 30, 'point': point})
        out.write_text(json.dumps([row['point'] for row in track]))
        print('extracted', out.name, flush=True)


def dataset():
    from backend.app.analytics import bounce_features
    X, y, groups = [], [], []
    for game, clip in train_clips():
        path = TRACKS / f'{game}_{clip.name}.json'
        if not path.exists():
            continue
        points = json.loads(path.read_text())
        xy = np.array([p if p is not None else [np.nan, np.nan] for p in points], float) * 2
        truth = labels(clip)[:len(xy)]
        bounce = np.zeros(len(xy), bool)
        for i, label in enumerate(truth):
            if label['status'] == 2:
                bounce[max(0, i-1):i+2] = True   # one-frame tolerance either side
        features = bounce_features(xy)
        keep = ~np.isnan(xy[:, 0])
        X.append(features[keep]); y.append(bounce[keep]); groups += [game] * int(keep.sum())
    return np.concatenate(X), np.concatenate(y), np.array(groups)


def train():
    import catboost as ctb
    X, y, groups = dataset()
    validation = groups == 'game7'
    print(f'{len(y)} frames, {int(y.sum())} bounce frames; validating on game7', flush=True)
    model = ctb.CatBoostClassifier(iterations=1500, depth=6, learning_rate=.05, auto_class_weights='Balanced',
                                   eval_metric='F1', random_seed=0, verbose=200)
    model.fit(X[~validation], y[~validation], eval_set=(X[validation], y[validation]), use_best_model=True)
    model.save_model(str(MODEL))
    print('saved', MODEL)


if __name__ == '__main__':
    {'extract': extract, 'train': train}[sys.argv[1]]()
