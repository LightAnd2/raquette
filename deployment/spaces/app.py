"""Hugging Face Spaces entry point for the Raquette API.

Third-party weights are downloaded at build time (download_models.py); Raquette's own models ship in this
repo. Startup fails if any is missing: results are never simulated.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ['ml/models/weights/tracknet_ball.pt', 'ml/models/weights/court_detector.pt',
            'ml/models/weights/bounce_detector.cbm', 'ml/models/weights/bounce_classifier.cbm',
            'ml/models/weights/serve_detector.pt', 'ml/models/weights/rally_classifier.pt',
            'yolov8n.pt', 'ml/train/pose_landmarker_lite.task']
missing = [path for path in REQUIRED if not (ROOT / path).exists()]
if missing:
    raise RuntimeError(f'Missing model weights: {missing}')

sys.path.insert(0, str(ROOT))
from backend.app.main import app  # noqa: E402,F401
