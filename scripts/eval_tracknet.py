"""Score the analytics pipeline against the TrackNet tennis dataset (broadcast angle).

Only games 8-10 are used: the original repo trained on the first 70% of frames
(games 1-7), so these games were never seen by the ball model.

  python scripts/eval_tracknet.py run      # build clip videos, run the pipeline, save results
  python scripts/eval_tracknet.py score    # compare saved results with the labels
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'datasets/tracknet/Dataset'
OUT = ROOT / 'datasets/tracknet/eval'
TEST_GAMES = ['game8', 'game9', 'game10']
FPS = 30


def clips():
    for game in TEST_GAMES:
        for clip in sorted((DATA / game).iterdir(), key=lambda p: int(p.name.replace('Clip', ''))):
            yield game, clip


def labels(clip):
    with open(clip / 'Label.csv') as file:
        return [{'visible': int(r['visibility'] or 0) in (1, 2),
                 'x': float(r['x-coordinate'] or 'nan'), 'y': float(r['y-coordinate'] or 'nan'),
                 'status': int(r['status'] or 0)} for r in csv.DictReader(file)]


def build_video(clip, path):
    frames = sorted(clip.glob('*.jpg'))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1280, 720))
    for frame in frames:
        writer.write(cv2.imread(str(frame)))
    writer.release()


def run():
    from backend.app.analytics import run_pipeline
    OUT.mkdir(parents=True, exist_ok=True)
    for game, clip in clips():
        name = f'{game}_{clip.name}'
        video, result = OUT / f'{name}.mp4', OUT / f'{name}.json'
        if result.exists():
            continue
        build_video(clip, video)
        result.write_text(json.dumps(run_pipeline(video, use_cache=False)))
        print('done', name, flush=True)


def match_events(predicted, truth, tolerance):
    """Greedy one-to-one matching of frame indices within a tolerance."""
    unused, hits = set(truth), 0
    for frame in predicted:
        best = min(unused, key=lambda t: abs(t-frame), default=None)
        if best is not None and abs(best-frame) <= tolerance:
            unused.discard(best)
            hits += 1
    return hits


def score():
    # Post-processing is recomputed from the saved raw tracks so it can be tuned without rerunning TrackNet.
    from backend.app.analytics import bounce_events, drop_orphans, drop_stationary, models, near_player_hits
    detector = models()[3]
    ball = {'visible': 0, 'found': 0, 'predicted': 0, 'correct': 0}
    bounce = {'truth': 0, 'predicted': 0, 'matched': 0}
    hit = {'truth': 0, 'predicted': 0, 'matched': 0}
    for game, clip in clips():
        result_path = OUT / f'{game}_{clip.name}.json'
        if not result_path.exists():
            continue
        result, truth = json.loads(result_path.read_text()), labels(clip)
        track = result['ball_track']
        result['bounces'] = [e for e in bounce_events(track, detector) if e['court'] is not None]
        for row, label in zip(track, truth):
            point = None if row['point'] is None else np.array(row['point']) * 2  # model pixels -> 1280x720
            close = point is not None and label['visible'] and np.hypot(*(point - [label['x'], label['y']])) <= 10
            ball['visible'] += label['visible']
            ball['found'] += bool(close)
            ball['predicted'] += point is not None
            ball['correct'] += bool(close)
        truth_bounces = [i for i, label in enumerate(truth) if label['status'] == 2]
        predicted_bounces = [event['frame'] for event in result['bounces']]
        bounce['truth'] += len(truth_bounces)
        bounce['predicted'] += len(predicted_bounces)
        bounce['matched'] += match_events(predicted_bounces, truth_bounces, 3)
        # Recomputed from the saved tracks so hit logic can be tuned without rerunning the pipeline.
        # Near-player hits: below y=300 of 720 (far-player contacts sit around 100-250; near serves around 340).
        truth_hits = [i for i, label in enumerate(truth) if label['status'] == 1 and label['y'] > 300]
        predicted_hits = [event['frame'] for event in near_player_hits(drop_orphans(drop_stationary([dict(row) for row in track])), result['player_track'], result['bounces'])]
        hit['truth'] += len(truth_hits)
        hit['predicted'] += len(predicted_hits)
        hit['matched'] += match_events(predicted_hits, truth_hits, 4)
    pct = lambda a, b: f'{100*a/max(b,1):.1f}%'
    print(f"ball   recall {pct(ball['found'], ball['visible'])}  precision {pct(ball['correct'], ball['predicted'])}"
          f"  (within 10 px at 1280x720, {ball['visible']} visible frames)")
    print(f"bounce recall {pct(bounce['matched'], bounce['truth'])}  precision {pct(bounce['matched'], bounce['predicted'])}"
          f"  (within 3 frames, {bounce['truth']} labelled bounces)")
    if hit['truth']:
        print(f"hit    recall {pct(hit['matched'], hit['truth'])}  precision {pct(hit['matched'], hit['predicted'])}"
              f"  (near player, within 4 frames, {hit['truth']} labelled hits)")


if __name__ == '__main__':
    {'run': run, 'score': score}[sys.argv[1]]()
