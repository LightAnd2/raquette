"""Cut a long broadcast video into rally clips: stretches where the main court camera is on screen.

Replays, player close-ups, crowd shots and graphics fail the court check (or put the court far from where
the main camera shows it) and are dropped.

  python scripts/split_rallies.py <video> [--out datasets/rallies] [--min-seconds 3]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app import analytics as A  # noqa: E402

SAMPLES_PER_SECOND = 2


def court_samples(path):
    """Court calibration twice a second (court keypoint model only: main broadcast views)."""
    court_model = A.models()[1]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / SAMPLES_PER_SECOND))
    samples, index = [], 0
    with torch.inference_mode():
        while True:
            if index % step:
                if not cap.grab():
                    break
            else:
                ok, image = cap.read()
                if not ok:
                    break
                frame = cv2.resize(image, (640, 360))
                calibration = A.court_prediction(court_model(A.tensor(frame))[0].sigmoid().cpu().numpy())
                samples.append({'time': index / fps, 'calibration': calibration})
            index += 1
    cap.release()
    return A.main_camera_only(samples, 640), fps


def segments(samples, min_seconds):
    """Runs of consecutive main-view samples; one missing sample (e.g. a player crossing a line) is bridged."""
    runs, start, misses = [], None, 0
    for k, sample in enumerate(samples):
        if sample['calibration']:
            start = k if start is None else start
            misses = 0
        elif start is not None:
            misses += 1
            if misses > 1:
                runs.append((start, k - misses))
                start, misses = None, 0
    if start is not None:
        runs.append((start, len(samples) - 1 - misses))
    # Broadcast cuts and crossfades fall between samples: start half a second late and stop at the last
    # main-view sample so clips never open on a dissolve or end on the next close-up.
    trimmed = [(samples[a]['time'] + 1 / SAMPLES_PER_SECOND, samples[b]['time']) for a, b in runs]
    return [(start, end) for start, end in trimmed if end - start >= min_seconds]


def write_clips(path, spans, fps, out):
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    index, written = 0, []
    for number, (start, end) in enumerate(spans, 1):
        clip = out / f'rally_{number:03d}.mp4'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'avc1'), fps, size)
        while index < round(start * fps):
            cap.grab()
            index += 1
        while index < round(end * fps):
            ok, image = cap.read()
            if not ok:
                break
            writer.write(image)
            index += 1
        writer.release()
        written.append({'clip': clip.name, 'start': round(start, 2), 'end': round(end, 2)})
    cap.release()
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--out', default=str(ROOT / 'datasets/rallies'))
    parser.add_argument('--min-seconds', type=float, default=3)
    args = parser.parse_args()
    video = Path(args.video)
    samples, fps = court_samples(video)
    spans = segments(samples, args.min_seconds)
    out = Path(args.out) / video.stem[:60].strip()
    clips = write_clips(video, spans, fps, out)
    (out / 'index.json').write_text(json.dumps({'source': video.name, 'clips': clips}, indent=1))
    covered = sum(c['end'] - c['start'] for c in clips)
    print(f'{len(clips)} rally clips, {covered/60:.1f} of {samples[-1]["time"]/60:.1f} minutes -> {out}')


if __name__ == '__main__':
    main()
