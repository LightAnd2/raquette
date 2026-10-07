"""Video analytics. No simulated results and no dependency on shot classification."""
from collections import deque
import itertools
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from ml.tennis_analysis.tracknet import BallTrackerNet
from ml.tennis_analysis.bounce_detector import BounceDetector

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'ml/models/weights'
VERSION = 'analytics-15'
DEVICE = os.environ.get('RAQUETTE_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
REFERENCE = np.float32([(286,561),(1379,561),(286,2935),(1379,2935),
                        (423,561),(423,2935),(1242,561),(1242,2935),
                        (423,1110),(1242,1110),(423,2386),(1242,2386),
                        (832,1110),(832,2386)])
COURT_POINTS = (REFERENCE - [286, 561]) / [1093, 2374] * [10.97, 23.77]
COURT = {'width': 10.97, 'height': 23.77, 'runoff': 8,
         'singlesInset': 1.37, 'serviceInset': 5.485, 'netY': 11.885}


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or not np.isfinite(fps) or fps <= 0 or count < 3:
            raise ValueError('This file could not be decoded as a video with at least three frames.')
        return {'width': frame.shape[1], 'height': frame.shape[0], 'fps': fps,
                'frame_count': count, 'duration': count / fps}
    finally:
        cap.release()


@lru_cache(maxsize=1)
def models():
    # Sequential jobs share model weights, but never share tracking state.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    loaded = []
    for name, inputs, outputs in [('tracknet_ball.pt',9,256), ('court_detector.pt',3,15)]:
        model = BallTrackerNet(input_channels=inputs, out_channels=outputs).eval()
        model.load_state_dict(torch.load(WEIGHTS / name, map_location='cpu', weights_only=True))
        loaded.append(model.to(DEVICE))
    if DEVICE != 'cpu':
        # Half precision ball model: same accuracy on held-out TrackNet frames (96.4% both), 16% faster.
        loaded[0] = loaded[0].half()
    else:
        # Channels-last on CPU: identical output on held-out TrackNet frames, about 1.4x faster.
        loaded[0] = loaded[0].to(memory_format=torch.channels_last)
    from ultralytics import YOLO
    person_weights = WEIGHTS / 'player_yolov8.pt'
    if not person_weights.exists():
        person_weights = ROOT / 'yolov8n.pt'
    if not person_weights.exists():
        raise RuntimeError('Player model weights are missing. No simulated analysis was generated.')
    detector = BounceDetector(str(WEIGHTS / 'bounce_detector.cbm'))
    classifier_path = WEIGHTS / 'bounce_classifier.cbm'
    if classifier_path.exists():
        # Trained on this pipeline's own tracks (scripts/train_bounce.py); held-out TrackNet games 8-10:
        # 82.5% recall / 94.1% precision at 0.8, against 72.7% / 94.1% for the original regressor.
        import catboost
        detector.classifier = catboost.CatBoostClassifier()
        detector.classifier.load_model(str(classifier_path))
        detector.classifier_threshold = .8
    return (*loaded, YOLO(str(person_weights)), detector)


def tensor(image):
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).to(DEVICE).float()[None] / 255


def ball_candidates(labels):
    # Class IDs ARE grayscale intensities in TrackNet's training targets.
    # Multiplying them by 255 wraps uint8 values and corrupts the heatmap.
    mask = (labels >= 128).astype(np.uint8)
    n, components, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 2 <= area <= 180 and max(w, h) <= 24:
            weights = labels[components == i].astype(float)
            ys, xs = np.where(components == i)
            candidates.append({'x': float(np.average(xs, weights=weights)),
                               'y': float(np.average(ys, weights=weights)),
                               'strength': float(weights.max() / 255)})
    return candidates


def fit_court(points):
    indices = [i for i, p in enumerate(points) if p is not None]
    if len(indices) < 8:
        return None
    src = COURT_POINTS[indices].astype(np.float32)
    dst = np.float32([points[i] for i in indices])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        return None
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 8 or inliers.mean() < .75:
        return None
    projected = cv2.perspectiveTransform(COURT_POINTS.astype(np.float32)[None], matrix)[0]
    quad = projected[[0,1,3,2]]
    if not cv2.isContourConvex(quad) or not 8000 < cv2.contourArea(quad) < 640*360*2:
        return None
    # Both baselines and both sidelines must have independent model support.
    supported = src[inliers]
    if np.ptp(supported[:,0]) < 8 or np.ptp(supported[:,1]) < 20:
        return None
    if quad[0,1] >= quad[3,1] or quad[1,1] >= quad[2,1]:
        return None
    error = float(np.median(np.linalg.norm(projected[indices][inliers] - dst[inliers], axis=1)))
    return {'image_to_court': np.linalg.inv(matrix).tolist(), 'quad': quad.tolist(),
            'reprojection_error': error, 'inliers': int(inliers.sum())}


def court_prediction(pred):
    points = []
    for heatmap in pred[:14]:
        peak = float(heatmap.max())
        if peak < .7:
            points.append(None)
            continue
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        mask = (heatmap >= max(.5, peak * .7)).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask)
        ys, xs = np.where(labels == labels[y,x])
        points.append([float(xs.mean()), float(ys.mean())])
    return fit_court(points)


_W, _L, _SI, _SV = 10.97, 23.77, 1.37, 5.485
COURT_LINES = [((0,0),(_W,0)),((0,_L),(_W,_L)),((0,0),(0,_L)),((_W,0),(_W,_L)),
               ((_SI,0),(_SI,_L)),((_W-_SI,0),(_W-_SI,_L)),((_SI,_SV),(_W-_SI,_SV)),
               ((_SI,_L-_SV),(_W-_SI,_L-_SV)),((_W/2,_SV),(_W/2,_L-_SV))]


def line_response(frame):
    # Painted lines are bright in every channel; clay and grass are dark in at least one.
    big = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2)).min(axis=2)
    return cv2.morphologyEx(big, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))).astype(np.float32)


def thick_line_response(frame):
    # Zoomed broadcast crops (e.g. Shorts) have near lines too thick for the thin-line filter.
    big = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2)).min(axis=2)
    return cv2.morphologyEx(big, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))).astype(np.float32)


# The near half alone fixes the homography, and it is the half the user clicked on.
# Raised objects (net tape, posts) project beyond the net, so they never match these lines.
NEAR_LINES = [((0,_L),(_W,_L)), ((_SI,_L-_SV),(_W-_SI,_L-_SV)),
              ((0,_L/2),(0,_L)), ((_SI,_L/2),(_SI,_L)), ((_W-_SI,_L/2),(_W-_SI,_L)), ((_W,_L/2),(_W,_L)),
              ((_W/2,_L/2),(_W/2,_L-_SV))]


def line_masks(response, thick=None):
    """Thin, isolated bright pixels; dense clutter (crowd, net mesh, logos) is removed."""
    binary = (response > 30).astype(np.float32)
    clean = (binary > 0) & (cv2.boxFilter(binary, -1, (21,21)) < .25)
    if thick is not None:
        # Thick lines: judged by edges (a line has two borders) and by not being a large filled area.
        wide = (thick > 30).astype(np.uint8)
        edges = cv2.Canny(wide*255, 50, 150).astype(np.float32) / 255
        clean |= (wide > 0) & (cv2.boxFilter(edges, -1, (21,21)) < .2) & (cv2.boxFilter(wide.astype(np.float32), -1, (41,41)) < .45)
    return clean.astype(np.uint8)


def near_court_lines(clean, matrix):
    """Painted lines inside the roughly placed near half-court, merged from Hough fragments."""
    region = np.float32([[-2,_L/2+.5],[_W+2,_L/2+.5],[_W+2,_L+2.5],[-2,_L+2.5]])
    polygon = cv2.perspectiveTransform(region[None], matrix)[0]
    mask = np.zeros(clean.shape, np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    found = cv2.HoughLinesP((clean & mask) * 255, 1, np.pi/720, 50, minLineLength=50, maxLineGap=20)
    clusters = []
    for x1, y1, x2, y2 in ([] if found is None else found[:,0]):
        angle = np.arctan2(y2-y1, x2-x1) % np.pi
        offset = np.array([-np.sin(angle), np.cos(angle)]) @ [x1, y1]
        for cluster in clusters:
            if abs((cluster['angle']-angle+np.pi/2) % np.pi - np.pi/2) < .03 and abs(cluster['offset']-offset) < 6:
                cluster['segments'].append(((x1,y1),(x2,y2)))
                break
        else:
            clusters.append({'angle':angle, 'offset':offset, 'segments':[((x1,y1),(x2,y2))]})
    lines = []
    for cluster in clusters:
        if sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in cluster['segments']) >= 80:
            lines.append(np.concatenate([np.array(a,float) + (np.array(b,float)-np.array(a,float)) * np.linspace(0,1,8)[:,None]
                                         for a, b in cluster['segments']]))
    return lines


def _image_line(matrix, a, b):
    p = cv2.perspectiveTransform(np.float32([[a,b]]), matrix)[0].astype(np.float64)
    direction = (p[1]-p[0]) / (np.linalg.norm(p[1]-p[0]) + 1e-9)
    return p[0], direction


def _solve_lines(pairs):
    """Image->court homography from (image points, model line) pairs: l^T H p = 0 is linear in H."""
    T = np.array([[1/640,0,-1],[0,1/640,-.5625],[0,0,1]])
    rows = []
    for points, (a, b) in pairs:
        a, b = np.array(a,float), np.array(b,float)
        line = np.cross([*a,1.], [*b,1.])
        for point in points:
            rows.append(np.kron(line, T @ [*point,1.]))
    _, _, vt = np.linalg.svd(np.array(rows))
    return vt[-1].reshape(3,3) @ T


def _residuals(court_from_image, points, a, b):
    """Pixel distance from image points to where the model line actually projects."""
    image_from_court = np.linalg.inv(court_from_image)
    origin, direction = _image_line(image_from_court, a, b)
    offset = points - origin
    return np.abs(offset[:,0]*direction[1] - offset[:,1]*direction[0])


def refine_court(frame, matrix):
    """Fit the court to whole painted near-half lines, starting from a rough court->image homography.

    Works at twice the model-image size; returns a court->model-image homography or None."""
    clean = line_masks(line_response(frame))
    current = np.diag([2.,2.,1.]) @ matrix
    lines = near_court_lines(clean, current)
    pairs = []
    for radius in (40, 20, 10):
        model = [(ab, *_image_line(current, *ab)) for ab in NEAR_LINES]
        pairs = []
        for points in lines:
            direction = (points[-1]-points[0]) / (np.linalg.norm(points[-1]-points[0]) + 1e-9)
            options = []
            for ab, origin, line_dir in model:
                if abs(direction[0]*line_dir[1] - direction[1]*line_dir[0]) > .1:
                    continue
                offset = points - origin
                distance = np.abs(offset[:,0]*line_dir[1] - offset[:,1]*line_dir[0]).max()
                if distance < radius:
                    options.append((distance, ab))
            if options:
                pairs.append((points, min(options, key=lambda o:o[0])[1]))
        while True:
            if len({ab for _, ab in pairs if ab[0][0] == ab[1][0]}) < 2 or len({ab for _, ab in pairs if ab[0][1] == ab[1][1]}) < 2:
                return None
            court_from_image = _solve_lines(pairs)
            errors = [_residuals(court_from_image, pts, *ab).mean() for pts, ab in pairs]
            worst = int(np.argmax(errors))
            if errors[worst] <= 3:
                break
            pairs.pop(worst)
        current = np.linalg.inv(court_from_image)
        current /= current[2,2]
    return np.diag([.5,.5,1.]) @ current


def candidate_lines(clean, density):
    """Long painted lines anywhere in the frame (twice model-image size), with fitted direction."""
    found = cv2.HoughLinesP(clean*255, 1, np.pi/720, 50, minLineLength=25, maxLineGap=40)
    clusters = []
    for x1, y1, x2, y2 in ([] if found is None else found[:,0]):
        angle = np.arctan2(y2-y1, x2-x1) % np.pi
        offset = np.array([-np.sin(angle), np.cos(angle)]) @ [x1, y1]
        for cluster in clusters:
            if abs((cluster['angle']-angle+np.pi/2) % np.pi - np.pi/2) < .03 and abs(cluster['offset']-offset) < 6:
                cluster['segments'].append(((x1,y1),(x2,y2)))
                break
        else:
            clusters.append({'angle':angle, 'offset':offset, 'segments':[((x1,y1),(x2,y2))]})
    lines = []
    for cluster in clusters:
        length = sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in cluster['segments'])
        if length < 100:
            continue
        points = np.concatenate([np.array(a,float) + (np.array(b,float)-np.array(a,float)) * np.linspace(0,1,8)[:,None]
                                 for a, b in cluster['segments']])
        vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
        # Court lines have bare ground on both sides; crowd and signage lines do not.
        normal, sides = np.array([-vy, vx]), []
        for distance in (-20, -12, 12, 20):
            probe = (points + normal*distance).astype(int)
            inside = (probe[:,0] >= 0) & (probe[:,0] < density.shape[1]) & (probe[:,1] >= 0) & (probe[:,1] < density.shape[0])
            if inside.any():
                sides.append(density[probe[inside,1], probe[inside,0]].mean())
        if sides and np.mean(sides) > .08:
            continue
        lines.append({'points':points, 'length':length, 'dir':np.array([vx,vy]), 'origin':np.array([x0,y0])})
    # Keep the longest of each family so short lengthwise lines are not crowded out by long crosswise ones
    # (net tape, signage text and duplicate baselines in zoomed footage).
    lines.sort(key=lambda line:-line['length'])
    is_flat = lambda line: abs(line['dir'][1]) < np.sin(np.radians(30))
    return [line for line in lines if is_flat(line)][:7] + [line for line in lines if not is_flat(line)][:6]


_MODEL_SEGMENTS = np.float32([[a, b] for a, b in COURT_LINES]).reshape(1,-1,2)


def court_fit_score(hits, matrix):
    """Unique painted pixels covered by the full court model at model-image size, minus half the uncovered ones."""
    projected = cv2.perspectiveTransform(_MODEL_SEGMENTS, np.diag([.5,.5,1.]) @ matrix)[0]
    if not np.isfinite(projected).all() or np.abs(projected).max() > 1e5:
        return -1e9, 0.
    canvas = np.zeros(hits.shape, np.uint8)
    for i in range(0, len(projected), 2):
        cv2.line(canvas, tuple(projected[i].astype(int)), tuple(projected[i+1].astype(int)), 1, 1)
    drawn = int(canvas.sum())
    on = int((canvas & hits).sum())
    return on - .5*(drawn-on), on / max(drawn, 1)


def _line_y(line, x):
    return line['origin'][1] + (x-line['origin'][0]) * line['dir'][1] / (line['dir'][0] + 1e-9)


def _line_x(line, y):
    return line['origin'][0] + (y-line['origin'][1]) * line['dir'][0] / (line['dir'][1] + 1e-9)


def search_court(lines, hits):
    """Label two detected crosswise lines and two lengthwise lines with court lines; keep the best-scoring court.

    Any two of the four crosswise lines may be the visible pair (portrait Shorts often crop the near
    sidelines but show the far half whole), taken top-to-bottom in image order."""
    crosswise = [((0,0),(_W,0)), ((_SI,_SV),(_W-_SI,_SV)), ((_SI,_L-_SV),(_W-_SI,_L-_SV)), ((0,_L),(_W,_L))]
    lengthwise = [((0,0),(0,_L)), ((_SI,0),(_SI,_L)), ((_W/2,_SV),(_W/2,_L-_SV)), ((_W-_SI,0),(_W-_SI,_L)), ((_W,0),(_W,_L))]
    angles = [np.degrees(np.arctan2(line['dir'][1], line['dir'][0])) % 180 for line in lines]
    flat = [i for i, angle in enumerate(angles) if min(angle, 180-angle) < 30]
    full_court = np.float32([[[0,0],[_W,0],[_W,_L],[0,_L]]])
    best = (-1e9, None)
    center, area = hits.shape[1], 4 * hits.size   # lines live at twice the model-image size
    for i, j in itertools.combinations(flat, 2):
        if abs((angles[i]-angles[j]+90) % 180 - 90) > 12:
            continue
        upper, lower = (i, j) if _line_y(lines[i], center) < _line_y(lines[j], center) else (j, i)
        for k, m in itertools.combinations(range(len(lines)), 2):
            if {k, m} & {i, j}:
                continue
            if min(abs(lines[lower]['dir'][0]*lines[q]['dir'][1] - lines[lower]['dir'][1]*lines[q]['dir'][0]) for q in (k, m)) < .25:
                continue
            row = _line_y(lines[lower], center)
            left, right = (k, m) if _line_x(lines[k], row) < _line_x(lines[m], row) else (m, k)
            for c_up, c_low in itertools.combinations(range(4), 2):
                for a, b in itertools.combinations(range(5), 2):
                    pairs = [(lines[upper]['points'], crosswise[c_up]), (lines[lower]['points'], crosswise[c_low]),
                             (lines[left]['points'], lengthwise[a]), (lines[right]['points'], lengthwise[b])]
                    try:
                        matrix = np.linalg.inv(_solve_lines(pairs))
                    except np.linalg.LinAlgError:
                        continue
                    if not np.isfinite(matrix).all():
                        continue
                    matrix /= matrix[2,2]
                    quad = cv2.perspectiveTransform(full_court, matrix)[0]
                    if (not cv2.isContourConvex(quad) or cv2.contourArea(quad) < .04*area
                            or quad[2,1] < quad[1,1] or quad[3,1] < quad[0,1] or quad[0,0] > quad[1,0] or quad[3,0] > quad[2,0]):
                        continue
                    score, _ = court_fit_score(hits, matrix)
                    if score > best[0]:
                        best = (score, matrix)
    return best[1]


def detect_court(frame, previous=None, search=True):
    """Court calibration from painted lines alone, for any camera angle. frame is the model-image BGR."""
    if frame.shape[0] > frame.shape[1]:
        # Portrait crops (Shorts) zoom in so far that wrong courts fit the few visible lines as well as the
        # right one (tested on AO Shorts: confident wrong fits). No court beats a wrong court.
        return None
    response = line_response(frame)
    binary = (response > 30).astype(np.float32)
    density = cv2.boxFilter(binary, -1, (15,15))
    clean = line_masks(response, thick_line_response(frame))
    hits = cv2.resize(cv2.dilate(clean, np.ones((5,5),np.uint8)), frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    attempts = []
    if previous is not None:
        attempts.append(lambda: refine_court(frame, previous))
    def full_search():
        if not search:
            return None
        found = search_court(candidate_lines(clean, density), hits)
        if found is None:
            return None
        rough = np.diag([.5,.5,1.]) @ found
        # Refinement uses near-half lines; when those are cropped away keep the searched court.
        refined = refine_court(frame, rough)
        return refined if refined is not None else rough
    attempts.append(full_search)
    for attempt in attempts:
        matrix = attempt()
        if matrix is None:
            continue
        score, ratio = court_fit_score(hits, np.diag([2.,2.,1.]) @ matrix)
        if ratio < .45 or score < 150:
            continue
        quad = cv2.perspectiveTransform(COURT_POINTS.astype(np.float32)[None], matrix)[0][[0,1,3,2]]
        return {'image_to_court': np.linalg.inv(matrix).tolist(), 'quad': quad.tolist(),
                'reprojection_error': None, 'inliers': None, 'source': 'lines', 'line_score': round(score,1),
                'line_coverage': round(ratio,3)}
    return None


# Bounce positions are good to roughly +-0.3-0.5 m (one rally ball was placed 0.24 m long when it was in),
# so anything within half a metre of a singles line is not called.
CALL_MARGIN = .5


def line_call(court):
    if court is None:
        return None
    x, y = court
    margin = min(x - _SI, _W - _SI - x, y, _L - y)
    return 'in' if margin > CALL_MARGIN else 'out' if margin < -CALL_MARGIN else 'close'


_CORNERS = np.float32([[[0,0],[_W,0],[_W,_L],[0,_L]]])


def same_view(calibration, previous, width, tolerance=.15):
    """Court corners within `tolerance` of the frame width of the previous main-camera court."""
    quad = cv2.perspectiveTransform(_CORNERS, np.linalg.inv(np.array(calibration['image_to_court'])))[0]
    return np.abs(quad - cv2.perspectiveTransform(_CORNERS, previous)[0]).max() <= tolerance * width


def main_camera_only(calibrations, width, tolerance=.15):
    """Broadcast main cameras barely move: drop courts far from the usual on-screen position.

    Replays and close-ups can fit a confident but wrong court; on a 2-minute AO broadcast every main-view
    court sat within 15 px of the median while replay fits were 300+ px away."""
    corners = np.float32([[[0,0],[_W,0],[_W,_L],[0,_L]]])
    placed = [(row, cv2.perspectiveTransform(corners, np.linalg.inv(np.array(row['calibration']['image_to_court'])))[0])
              for row in calibrations if row['calibration']]
    if len(placed) < 5:
        return calibrations
    usual = np.median([quad for _, quad in placed], axis=0)
    for row, quad in placed:
        if np.abs(quad - usual).max() > tolerance * width:
            row['calibration'] = None
    return calibrations


def map_positions(result, calibration_at):
    for row in result['ball_track']:
        row['court'] = project(row['point'],calibration_at(row['time']))
    for row in result['player_track']:
        for player in row['players']:
            x1,y1,x2,y2 = player['box']
            player['court'] = project([(x1+x2)/2,y2],calibration_at(row['time']))
    for event in result['bounces']:
        event['court'] = project(event['point'],calibration_at(event['time']))


def project(point, calibration):
    if point is None or calibration is None:
        return None
    matrix = np.array(calibration['image_to_court'])
    result = matrix @ np.array([*point, 1.])
    if not np.isfinite(result).all() or abs(result[2]) < 1e-8:
        return None
    xy = result[:2] / result[2]
    # Reject implausible projections, never clamp them onto court boundaries.
    if not (-8 <= xy[0] <= 19 and -12 <= xy[1] <= 36):
        return None
    return [round(float(v), 3) for v in xy]


def choose_ball(candidates, history, timestamp):
    if not candidates:
        return None
    recent = [p for p in history[-5:] if p['point'] is not None and timestamp - p['time'] < .15]
    if recent:
        last = recent[-1]['point']
        ranked = sorted(candidates, key=lambda p: np.linalg.norm(np.array([p['x'],p['y']])-last))
        chosen = ranked[0]
        # Speeds in model-image pixels/second, independent of source resolution.
        if np.linalg.norm(np.array([chosen['x'],chosen['y']])-last) > 45 + 900*(timestamp-recent[-1]['time']):
            return None
    elif len(candidates) == 1:
        chosen = candidates[0]
    else:
        return None
    return [chosen['x'], chosen['y']]


def bounce_features(xy):
    """Per-frame motion features from a 30 fps ball track (N x 2, 1280x720 pixels, NaN = missing).

    Shared by training (scripts/train_bounce.py) and inference so both see identical inputs."""
    xy = np.asarray(xy, float)
    n = len(xy)
    def shifted(k):
        out = np.full_like(xy, np.nan)
        if k > 0:
            out[k:] = xy[:-k]
        elif k < 0:
            out[:k] = xy[-k:]
        else:
            out[:] = xy
        return out
    columns = []
    for k in range(1, 6):
        columns += [(shifted(k) - xy), (shifted(-k) - xy)]          # displacement to past / future frames
    before = (xy - shifted(3)) / 3                                   # mean velocity in / out
    after = (shifted(-3) - xy) / 3
    accel = shifted(-1) - 2*xy + shifted(1)
    speed_in, speed_out = np.hypot(*before.T), np.hypot(*after.T)
    cos_turn = (before*after).sum(1) / (speed_in*speed_out + 1e-6)
    extra = np.column_stack([speed_in, speed_out, speed_out/(speed_in+1e-6), cos_turn,
                             before[:,1]*after[:,1], xy[:,1]/720, np.isnan(shifted(1)[:,0]), np.isnan(shifted(-1)[:,0])])
    features = np.column_stack(columns + [before, after, accel, extra])
    return np.where(np.isnan(xy[:,:1]), np.nan, features) if n else features.reshape(0, features.shape[1])


def bounce_events(track, detector):
    if len(track) < 5:
        return []
    times = np.array([p['time'] for p in track])
    xy = np.array([p['point'] if p['point'] is not None else [np.nan,np.nan] for p in track])
    # The bounce regressor was trained on 1280x720, ~30fps coordinates.
    # Resample time, not repeated frame indices; do not bridge tracking gaps.
    grid = np.arange(times[0], times[-1] + 1e-8, 1/30)
    sampled = np.full((len(grid),2), np.nan)
    for i, t in enumerate(grid):
        right = np.searchsorted(times, t)
        if right < len(times) and abs(times[right]-t) < .002:
            sampled[i] = xy[right] * 2
        elif 0 < right < len(times) and times[right]-times[right-1] <= .065:
            a, b = xy[right-1], xy[right]
            sampled[i] = (a + (b-a)*(t-times[right-1])/(times[right]-times[right-1])) * 2
    if getattr(detector, 'classifier', None) is not None:
        keep = np.where(np.isfinite(sampled[:,0]))[0]
        if not len(keep):
            return []
        scores = detector.classifier.predict_proba(bounce_features(sampled)[keep])[:,1]
        candidates = [(int(f),float(s)) for f,s in zip(keep,scores) if s > detector.classifier_threshold]
    else:
        x = [None if not np.isfinite(p[0]) else p[0] for p in sampled]
        y = [None if not np.isfinite(p[1]) else p[1] for p in sampled]
        features, frames = detector.prepare_features(x, y)
        if features.empty:
            return []
        scores = detector.model.predict(features)
        candidates = [(f,float(s)) for f,s in zip(frames,scores) if s > detector.threshold]
    # Merge neighboring positive frames in time, not positions in a filtered list.
    groups = []
    for f, score in candidates:
        if not groups or f - groups[-1][-1][0] > 1:
            groups.append([])
        groups[-1].append((f, score))
    events = []
    for group in groups:
        frame, score = max(group, key=lambda p:p[1])
        source = min(track, key=lambda p:abs(p['time']-grid[frame]))
        events.append({'id':len(events)+1, 'time':round(float(grid[frame]),4),
                       'frame':source['frame'], 'point':(sampled[frame]/2).tolist(),
                       'model_score':round(score,4), 'court':source.get('court')})
    return events


def drop_stationary(track, span=8, speed=1.5, count=5):
    """A ball in play never creeps: remove runs of detections moving under `speed` px/frame that last
    at least `span` frames over `count` or more detections (logos, painted letters)."""
    visible = [i for i, row in enumerate(track) if row['point'] is not None]
    slow = [False] + [np.hypot(*(np.array(track[b]['point']) - track[a]['point'])) / (track[b]['frame'] - track[a]['frame']) < speed
                      for a, b in zip(visible, visible[1:])]
    k = 0
    while k < len(visible):
        end = k
        while end + 1 < len(visible) and slow[end + 1]:
            end += 1
        if end - k + 1 >= count and track[visible[end]]['frame'] - track[visible[k]]['frame'] >= span:
            for i in visible[k:end+1]:
                track[i]['point'] = None
        k = end + 1
    return track


def drop_orphans(track, frames=5, distance=40.):
    """Remove detections with no consistent neighbour within a few frames: a ball in flight forms a path."""
    visible = [i for i, row in enumerate(track) if row['point'] is not None]
    orphans = []
    for k, i in enumerate(visible):
        neighbours = [j for j in visible[max(0,k-2):k] + visible[k+1:k+3]
                      if abs(track[j]['frame'] - track[i]['frame']) <= frames
                      and np.hypot(*(np.array(track[j]['point']) - track[i]['point'])) <= distance * abs(track[j]['frame'] - track[i]['frame'])]
        if not neighbours:
            orphans.append(i)
    for i in orphans:
        track[i]['point'] = None
    return track


def near_player_hits(track, people, bounces):
    """Near-player contacts: the ball stops coming toward the camera and heads back, beside the player.

    Both supported angles film the near player from behind, so an incoming ball moves down the
    image until it is struck. Turning points at a detected bounce are bounces, not hits."""
    samples = [(row['time'], row['players'][0]['box']) for row in people if row['players']]
    if not samples:
        return []
    sample_times = np.array([time for time, _ in samples])
    times = np.array([row['time'] for row in track])
    points = np.array([row['point'] if row['point'] is not None else [np.nan, np.nan] for row in track], float)
    visible = np.where(~np.isnan(points[:,1]))[0]
    bounce_times = np.array([event['time'] for event in bounces])
    hits = []
    for i in visible:
        before = visible[(times[visible] >= times[i]-.25) & (times[visible] < times[i])]
        after = visible[(times[visible] > times[i]) & (times[visible] <= times[i]+.25)]
        if len(before) < 2 or len(after) < 2:
            continue
        y = points[i,1]
        if y < points[before,1].max() or y < points[after,1].max():
            continue
        if y - points[before,1].min() < 8 or y - points[after,1].min() < 8:  # a real reversal, not jitter
            continue
        if len(bounce_times) and np.abs(bounce_times - times[i]).min() < .15:
            continue
        nearest = int(np.abs(sample_times - times[i]).argmin())
        if abs(sample_times[nearest] - times[i]) > .4:
            continue
        x1, y1, x2, y2 = samples[nearest][1]
        w, h = x2-x1, y2-y1
        if not (x1-1.5*w <= points[i,0] <= x2+1.5*w and y1-.6*h <= y <= y2+.2*h):
            continue
        if hits and times[i] - hits[-1]['time'] < .6:
            continue
        hits.append({'time':round(float(times[i]),4), 'frame':track[i]['frame'], 'point':points[i].tolist(),
                     'player_box':[float(v) for v in samples[nearest][1]]})
    # Contacts hidden in a tracking gap: the ball approached the near player before it and left after it.
    for before, after in zip(visible, visible[1:]):
        gap = times[after] - times[before]
        if not .1 < gap <= 1.:
            continue
        incoming = visible[(times[visible] >= times[before]-.25) & (times[visible] <= times[before])]
        outgoing = visible[(times[visible] >= times[after]) & (times[visible] <= times[after]+.25)]
        if len(incoming) < 3 or len(outgoing) < 3:
            continue
        speed_in = np.polyfit(times[incoming], points[incoming,1], 1)[0]
        speed_out = np.polyfit(times[outgoing], points[outgoing,1], 1)[0]
        # Toward the camera before the gap; afterwards heading away, or already back up the court.
        if speed_in <= 20 or (speed_out >= -20 and points[after,1] > points[before,1] - 30):
            continue
        if len(bounce_times) and np.any((bounce_times > times[before]) & (bounce_times < times[after])):
            continue
        nearest = int(np.abs(sample_times - times[before]).argmin())
        x1, y1, x2, y2 = samples[nearest][1]
        w, h = x2-x1, y2-y1
        if not (x1-3*w <= points[before,0] <= x2+3*w and points[before,1] >= y1-1.5*h):
            continue
        velocity = np.array([np.polyfit(times[incoming], points[incoming,k], 1)[0] for k in (0,1)])
        reach = np.hypot((x1+x2)/2 - points[before,0], (y1+y2)/2 - points[before,1])
        contact = times[before] + np.clip(reach / max(np.hypot(*velocity), 1e-6), 0, gap)
        if any(abs(hit['time'] - contact) < .6 for hit in hits):
            continue
        frame = track[int(np.abs(times - contact).argmin())]['frame']
        hits.append({'time':round(float(contact),4), 'frame':frame, 'point':[float(np.clip(points[before,0], x1-w, x2+w)), float((y1+y2)/2)],
                     'player_box':[float(v) for v in samples[nearest][1]], 'inferred':True})
    hits.sort(key=lambda hit: hit['time'])
    for number, hit in enumerate(hits, 1):
        hit['id'] = number
    return hits


def pick_near_player(samples, height=360):
    # Only the near-side player is analysed: the tallest person standing in the
    # lower half of the frame. Per sample, so swing poses never split a track.
    for sample in samples:
        boxes = [b for b in sample.pop('boxes') if b['box'][3] > height/2 and b['box'][3]-b['box'][1] > height/9]
        near = max(boxes, key=lambda b:b['box'][3]-b['box'][1], default=None)
        sample['players'] = [{**near, 'id':1}] if near else []
    return samples


def cache_key(path):
    digest = hashlib.sha256(VERSION.encode())
    with open(path, 'rb') as file:
        for chunk in iter(lambda:file.read(1024*1024), b''):
            digest.update(chunk)
    for weight in [WEIGHTS/'tracknet_ball.pt',WEIGHTS/'court_detector.pt',WEIGHTS/'bounce_detector.cbm',WEIGHTS/'bounce_classifier.cbm',ROOT/'yolov8n.pt']:
        stat = weight.stat()
        digest.update(f'{weight.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()


def run_pipeline(video_path, progress_cb=lambda **kwargs:None, mode='singles', player_names=None, use_cache=True, handedness=None):
    if mode != 'singles':
        raise ValueError('The current analytics pipeline supports singles footage only.')
    info = video_info(video_path)
    cached = ROOT / 'backend/uploads/cache' / f'{cache_key(video_path)}.json'
    if use_cache and cached.exists():
        result = json.loads(cached.read_text())
        result['cache_hit'] = True
        if handedness:
            from .shots import relabel
            result['shots'] = relabel(result.get('shots', []), handedness)
        result['player_names'] = (player_names or ['Near player'])[-1:]
        return result
    progress_cb(progress=1, stage='Loading models')
    ball_model, court_model, person_model, bounce_model = models()
    # Models run at the clip's own shape: portrait clips (e.g. YouTube Shorts) are not stretched.
    size = (640,360) if info['width'] >= info['height'] else (360,640)
    cap = cv2.VideoCapture(str(video_path))
    frames = deque(maxlen=3)
    track, people, calibrations = [], [], []
    last_search, search_wait = -np.inf, 5
    started = time.monotonic()
    last_time = -1.
    try:
        with torch.inference_mode():
            frame_index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not np.isfinite(timestamp) or timestamp <= last_time:
                    timestamp = frame_index / info['fps']
                last_time = timestamp
                resized = cv2.resize(image, size)
                frames.append(resized)
                if frame_index % max(1,round(info['fps'])) == 0:
                    calibration = court_prediction(court_model(tensor(resized))[0].sigmoid().cpu().numpy())
                    if calibration is None:
                        # Low or unusual angles: fit the painted lines directly.
                        previous = next((np.linalg.inv(np.array(c['calibration']['image_to_court']))
                                         for c in reversed(calibrations) if c['calibration']), None)
                        # The full line search costs seconds; during replays and cutaways it would run every
                        # sample for nothing, so back off (5, 10, 20, 30 s) while it keeps failing.
                        search = timestamp - last_search >= search_wait
                        calibration = detect_court(resized, previous, search)
                        if calibration is not None and previous is not None and not same_view(calibration, previous, size[0]):
                            calibration = None   # a replay or close-up, not the main camera
                        if search and calibration is None:
                            last_search, search_wait = timestamp, min(30, search_wait*2)
                    if calibration is not None:
                        search_wait = 5
                    calibrations.append({'time':timestamp, 'calibration':calibration})
                if frame_index % max(1,round(info['fps']/6)) == 0:
                    result = person_model.predict(image, classes=[0],imgsz=640,conf=.4,verbose=False,device=DEVICE)[0]
                    boxes = []
                    for box, score in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                        box = box / [info['width']/size[0],info['height']/size[1],info['width']/size[0],info['height']/size[1]]
                        if box[3] > size[1]*.4:
                            boxes.append({'box':box.tolist(), 'score':float(score)})
                    people.append({'time':timestamp,'frame':frame_index,'boxes':boxes})
                point, candidates = None, []
                if len(frames) == 3:
                    stacked = tensor(np.concatenate(list(frames)[::-1],axis=2)).to(next(ball_model.parameters()).dtype)
                    if DEVICE == 'cpu':
                        stacked = stacked.contiguous(memory_format=torch.channels_last)
                    labels = ball_model(stacked)[0].argmax(0).cpu().numpy().astype(np.uint8)
                    candidates = ball_candidates(labels)
                    point = choose_ball(candidates,track,timestamp)
                track.append({'frame':frame_index,'time':round(timestamp,6),'point':point,'candidates':candidates})
                if frame_index % 5 == 0:
                    progress_cb(progress=min(94,2+92*(frame_index+1)/info['frame_count']),
                                stage='Tracking ball and players', processed_frames=frame_index+1,
                                total_frames=info['frame_count'])
                frame_index += 1
    finally:
        cap.release()
    if len(track) < info['frame_count']*.95:
        raise ValueError('Video decoding ended early. Export a standard H.264 MP4 and try again.')
    progress_cb(progress=96,stage='Evaluating trajectory')
    calibrations = main_camera_only(calibrations, size[0])
    people = pick_near_player(people, size[1])
    def calibration_at(timestamp):
        nearest = min(calibrations,key=lambda p:abs(p['time']-timestamp))
        return nearest['calibration'] if abs(nearest['time']-timestamp) < 1.1 else None
    events = bounce_events(track,bounce_model)
    coverage = sum(row['point'] is not None for row in track)/len(track)
    valid_courts = sum(row['calibration'] is not None for row in calibrations)
    # Only flag what changes how far the results can be trusted. Broadcast ball tracks have normal gaps
    # (out of frame, behind players): 66% coverage still gave every shot and bounce on the AO 2017 test clip.
    warnings = []
    if size[1] > size[0]:
        warnings.append("Vertical clip, so there's no court map. Ball, player and shots are still tracked.")
    elif not valid_courts:
        warnings.append("Couldn't find the court in this clip, so nothing is placed on the court map.")
    if coverage < .5:
        warnings.append('The ball was hard to follow in this clip, so some bounces and shots may be missing.')
    result = {'schema_version':1,'pipeline_version':VERSION,'video':info,'coordinate_space':{'width':size[0],'height':size[1]},
              'ball_track':track,'player_track':people,'calibrations':calibrations,'court_model':COURT,
              'bounces':events,
              'quality':{'ball_coverage':coverage,'calibrated_samples':valid_courts,'court_samples':len(calibrations)},
              'warnings':warnings,'player_names':(player_names or ['Near player'])[-1:],
              'processing_seconds':round(time.monotonic()-started,2),'cache_hit':False,
              'models':{'ball':'TrackNet','court':'TennisCourtDetector','bounce':'CatBoost','players':'YOLOv8'}}
    map_positions(result, calibration_at)
    # A bounce is on the ground: with a known court it must map onto the court surface.
    result['bounces'] = [event for event in result['bounces']
                         if event['court'] is not None or calibration_at(event['time']) is None]
    for number, event in enumerate(result['bounces'], 1):
        event['id'] = number
        event['call'] = line_call(event['court'])
    # Creeping false detections only mislead contact finding; ball and bounce output keep every point.
    result['hits'] = near_player_hits(drop_orphans(drop_stationary([dict(row) for row in track])), people, result['bounces'])
    if valid_courts:
        # Broadcasts cut to replays and close-ups; a rally shot is only counted while the court view is on screen.
        result['hits'] = [hit for hit in result['hits'] if calibration_at(hit['time']) is not None]
        for number, hit in enumerate(result['hits'], 1):
            hit['id'] = number
    progress_cb(progress=98, stage='Classifying shots')
    from .shots import classify_shots
    result['shots'] = classify_shots(video_path, result, result['hits'], info, size, handedness)
    cached.parent.mkdir(parents=True,exist_ok=True)
    temporary = cached.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,allow_nan=False))
    temporary.replace(cached)
    return result
