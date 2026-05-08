# Raquette

### AI Tennis Shot Identifier

Upload match footage. Get a timestamped breakdown of every shot — type, player, and moment of contact.

**[Explore the code »](https://github.com/LightAnd2/raquette)**&nbsp;&nbsp;·&nbsp;&nbsp;**[View Live App](https://raquette.vercel.app)**&nbsp;&nbsp;·&nbsp;&nbsp;**[Report Bug](https://github.com/LightAnd2/raquette/issues)**&nbsp;&nbsp;·&nbsp;&nbsp;**[Request Feature](https://github.com/LightAnd2/raquette/issues)**

---

## Table of Contents

- [About](#about)
- [How It Works](#how-it-works)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contact](#contact)

---

## About

Raquette is a full-stack AI application that identifies tennis shot types from match footage using a three-model ML pipeline. Upload an MP4, enter player names, and get a shot-by-shot timeline — every forehand, backhand, serve, volley, smash, slice, and return, attributed to the right player with timestamps.

**Why I built it**

- Coaching analytics tools cost thousands and require proprietary hardware
- Public match video has no structured shot data attached to it
- I wanted a pipeline that could turn any match recording into structured insight

**Features**

- Singles and doubles mode (2 or 4 players)
- Optional player name input — results show real names instead of P1/P2
- Player re-identification — players are tracked consistently across frames; ball boys and spectators are filtered out automatically
- Live shot feed during processing — see shots appear in real time
- Shot classification across 8 types: Forehand, Backhand, Serve, Return, Volley, Smash, Slice, Tweener
- Shot distribution chart by type and by player
- Full rally timeline with timestamps — click any shot for detail
- Fully deployed — frontend on Vercel, backend on Hugging Face Spaces

---

## How It Works

Three models run in sequence on every processed frame:

| Step | Model | What it does |
|------|-------|-------------|
| 01 | **YOLOv8n** | Detects players each frame, filters out ball boys and spectators by bounding box size and confidence |
| 02 | **MediaPipe Pose** | Extracts 33 body landmarks per player — shoulder rotation, hip alignment, wrist angle — the full biomechanical signature of each swing |
| 03 | **Temporal CNN** | A 1D convolutional network slides over the pose sequence to classify shot type from wrist velocity and body position |

A centroid-based re-identification tracker ensures each player is assigned a consistent identity across all frames. New detections are matched to the nearest known player slot — if no match is found within a distance threshold, the detection is discarded.

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
- [YOLOv8n](https://docs.ultralytics.com/) — player detection
- [MediaPipe](https://mediapipe.dev/) — pose estimation
- [PyTorch](https://pytorch.org/) — shot classifier (1D temporal CNN, trained on ~768 labeled samples)

**Infrastructure**
- [Vercel](https://vercel.com/) — frontend hosting
- [Hugging Face Spaces](https://huggingface.co/spaces) — backend + model hosting (Docker)
- [Hugging Face Hub](https://huggingface.co/LightAnd2/raquette-weights) — model weight storage

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
2. Select **Singles** or **Doubles** mode
3. Optionally enter player names (e.g. "Federer", "Nadal")
4. Drag and drop an MP4 of a tennis match, or click to browse
5. Watch the live shot feed as frames are processed
6. View the full results: shot breakdown by type and player, rally timeline with timestamps

**Tips**
- Broadcast or behind-the-baseline angles work best
- Shorter clips (30s–2min) process faster on the free-tier CPU backend
- The classifier is most accurate on clear full-swing groundstrokes

---

## Project Structure

```
raquette/
├── frontend/                  # React + Vite app
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx    # Upload + mode toggle + demo
│   │   │   ├── Analysis.jsx   # Live shot feed during processing
│   │   │   └── Results.jsx    # Shot breakdown + player stats + timeline
│   │   ├── components/
│   │   │   ├── SampleAnalysis.jsx
│   │   │   └── RallyTimeline.jsx
│   │   └── api.js
│   └── public/
│       └── demo.mp4
│
├── backend/                   # FastAPI server
│   └── app/
│       └── pipeline_worker.py # ML pipeline + PlayerTracker re-ID
│
├── ml/                        # ML models and utilities
│   ├── models/
│   │   ├── shot_classifier.py # Temporal CNN (ShotCNNFlat architecture)
│   │   └── weights/           # .pt files (downloaded at runtime from HF Hub)
│   └── utils/
│       └── video.py
│
├── hf-space/                  # Hugging Face Spaces deployment
│   ├── Dockerfile
│   ├── app.py                 # FastAPI entry point + weight bootstrap
│   └── requirements.txt
│
└── tennis_clips/
    ├── label.py               # Manual labeling tool for training data
    └── labels.csv             # ~912 labeled shot samples
```

---

## Roadmap

- [ ] Retrain classifier to 80%+ accuracy (more serve/return/smash samples)
- [ ] Lower frame skip on GPU backend for denser shot detection
- [ ] Match-level summary across multiple rallies
- [ ] Export results as PDF report

---

## Contact

**Andrew Koja**

- GitHub: [LightAnd2](https://github.com/LightAnd2)
- LinkedIn: [linkedin.com/in/andrewkoja](https://linkedin.com/in/andrewkoja)
- Project: [github.com/LightAnd2/raquette](https://github.com/LightAnd2/raquette)
