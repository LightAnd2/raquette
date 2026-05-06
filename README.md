# Raquette

### AI Tennis Shot Analysis

Upload match footage. Get a full biomechanical breakdown — every shot, every rally, every player.

**[Explore the code »](https://github.com/LightAnd2/raquette)**&nbsp;&nbsp;·&nbsp;&nbsp;**[View Live App](https://raquette.vercel.app)**&nbsp;&nbsp;·&nbsp;&nbsp;**[Report Bug](https://github.com/LightAnd2/raquette/issues)**&nbsp;&nbsp;·&nbsp;&nbsp;**[Request Feature](https://github.com/LightAnd2/raquette/issues)**

---

## Table of Contents

- [About](#about)
- [How It Works](#how-it-works)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contact](#contact)

---

## About

Raquette is a full-stack AI app that analyses tennis match footage through a 4-model ML pipeline. Upload an MP4, and within minutes you get a shot-by-shot breakdown: type, speed, player attribution, court position, and rally timeline — all visualised in a clean interface.

**Why I built it**

- Coaching analytics tools cost thousands and require proprietary hardware
- Publicly available video has no structured shot data attached to it
- I wanted a pipeline that could turn any match recording into structured, queryable insight

**Features**

- Drag-and-drop video upload with live progress feed
- Real-time frame streaming while analysis runs
- Shot classification across 7 types: Forehand, Backhand, Serve, Return, Volley, Smash, Slice
- Ball speed estimation per shot
- Court heatmap of shot landing positions
- Rally timeline with per-shot detail panel
- Shot distribution donut chart
- Player attribution (P1 vs P2) per shot
- Fully deployed — frontend on Vercel, backend on Hugging Face Spaces

---

## How It Works

Four models run in sequence on every frame:

| Step | Model | What it does |
|------|-------|-------------|
| 01 | **YOLOv8** | Detects and tracks both players across frames |
| 02 | **TrackNet V2** | Tracks the ball at high speed, predicting through occlusion |
| 03 | **MediaPipe** | Extracts 33 body landmarks at contact — shoulder, hip, wrist angles |
| 04 | **Temporal CNN** | Classifies shot type from a sliding window of pose sequences |

---

## Built With

**Frontend**
- [React](https://react.dev/) + [Vite](https://vitejs.dev/)
- [Tailwind CSS v4](https://tailwindcss.com/)
- [Framer Motion](https://www.framer.com/motion/)
- [Recharts](https://recharts.org/)
- [React Router](https://reactrouter.com/)

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/)
- [uvicorn](https://www.uvicorn.org/)

**ML Pipeline**
- [YOLOv8](https://docs.ultralytics.com/) — player detection
- [TrackNet V2](https://github.com/yastrebksv/TrackNet) — ball tracking
- [MediaPipe](https://mediapipe.dev/) — pose estimation
- [PyTorch](https://pytorch.org/) — shot classifier (1D-CNN)

**Infrastructure**
- [Vercel](https://vercel.com/) — frontend hosting
- [Hugging Face Spaces](https://huggingface.co/spaces) — backend + model hosting (Docker)

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.11+

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/LightAnd2/raquette.git
cd raquette
```

2. **Set up the Python environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

3. **Install frontend dependencies**

```bash
cd frontend && npm install
```

4. **Start both servers**

```bash
cd .. && ./dev.sh
```

5. **Open the app**

```
http://localhost:5173
```

> The backend runs on port 8000, frontend on 5173. `dev.sh` starts both.

---

## Usage

1. Go to [raquette.vercel.app](https://raquette.vercel.app) or run locally
2. Drag and drop an MP4 of a tennis match (or click to browse)
3. Watch the live analysis feed as frames are processed
4. View the full results: shot breakdown, court heatmap, rally timeline

**Tips**
- Footage shot from behind the baseline or a broadcast angle works best
- Shorter clips (30s–2min) are faster to process on the free-tier backend
- The shot classifier is most accurate on full-swing groundstrokes

---

## Project Structure

```
raquette/
├── frontend/                  # React + Vite app
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx    # Upload + demo + pipeline explainer
│   │   │   ├── Analysis.jsx   # Live progress feed
│   │   │   └── Results.jsx    # Shot breakdown + heatmap + timeline
│   │   ├── components/
│   │   │   ├── CourtHeatmap.jsx
│   │   │   └── RallyTimeline.jsx
│   │   └── api.js             # Fetch wrapper (reads VITE_API_URL)
│   └── public/
│       └── demo.mp4
│
├── backend/                   # FastAPI server
│   └── app/
│       ├── main.py            # Upload, job polling, results endpoints
│       └── pipeline_worker.py # Orchestrates ML pipeline per job
│
├── ml/                        # ML models and utilities
│   ├── models/
│   │   ├── tracknet.py        # TrackNet V2 architecture
│   │   ├── shot_classifier.py # 1D-CNN over pose sequences
│   │   └── weights/           # .pt files (not in git — hosted on HF)
│   ├── pipeline.py            # Full pipeline orchestration
│   └── utils/
│       ├── video.py           # Frame iteration, overlay, JPEG encoding
│       └── court.py           # Perspective homography
│
├── notebooks/
│   └── train_raquette.ipynb   # Colab training notebook
│
├── deployment/
│   └── spaces/                # Hugging Face Spaces config
│
├── docs/
│   └── images/                # README assets
│
└── dev.sh                     # Local dev launcher
```

---

## Roadmap

- [ ] Retrain shot classifier on real match footage
- [ ] Per-player stats split (P1 vs P2 breakdown page)
- [ ] Match-level summary across multiple rallies
- [ ] Export results as PDF report
- [ ] GPU-backed inference for faster processing

---

## Contact

**Andrew Koja**

- GitHub: [LightAnd2](https://github.com/LightAnd2)
- LinkedIn: [linkedin.com/in/andrewkoja](https://linkedin.com/in/andrewkoja)
- Project: [github.com/LightAnd2/raquette](https://github.com/LightAnd2/raquette)
