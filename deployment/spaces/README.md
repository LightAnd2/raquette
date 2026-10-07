---
title: Raquette
emoji: 🎾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Raquette: Tennis Rally Analysis API

Backend for Raquette: upload a broadcast rally and get the court, ball track, bounces with in/out calls,
and the near player's serves, forehands and backhands.

The React frontend is deployed separately on Vercel and calls this API. Third-party model weights are downloaded at
build time; Raquette's own models ship in this repo.

## Accuracy

Measured on TrackNet dataset games 8-10, held out from training (5,675 broadcast frames).

| | Found | Correct |
|---|---|---|
| Ball position (within 10 px at 1280x720) | 94.2% | 99.7% |
| Bounces (within 3 frames) | 82.5% | 96.2% |
| Near-player hits (within 4 frames) | 91.8% | 97.1% |

Shot types were checked by hand on 52 near-player shots from Australian and US Open broadcasts:
forehand/backhand 46 of 48 correct, serves 4 of 4 with no false serves.

Supported: landscape TV broadcast footage, singles, near-side player.

## Credits

- TrackNet: Huang et al., "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications" (2019); weights from the PyTorch implementation by yastrebksv.
- TennisCourtDetector and the original bounce regressor: yastrebksv.
- YOLOv8: Ultralytics.
- MediaPipe Pose: Google.
