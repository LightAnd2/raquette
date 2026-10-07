"""Download the public third-party weights (Google Drive) into ml/models/weights.

Run once after cloning: python scripts/download_weights.py
Raquette's own small models are committed to the repo.
"""
from pathlib import Path

import gdown

WEIGHTS = Path(__file__).resolve().parents[1] / 'ml' / 'models' / 'weights'
MODELS = {
    'tracknet_ball.pt': '1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl',     # TrackNet ball model
    'court_detector.pt': '1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG',    # TennisCourtDetector
    'bounce_detector.cbm': '1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ',  # original bounce regressor (fallback)
}

if __name__ == '__main__':
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for name, file_id in MODELS.items():
        target = WEIGHTS / name
        if not (target.exists() and target.stat().st_size > 1000):
            if not gdown.download(f'https://drive.google.com/uc?id={file_id}', str(target), quiet=False):
                raise SystemExit(f'Could not download {name}')
        print(f'[weights] {name}: {target.stat().st_size // 1024} KB')
