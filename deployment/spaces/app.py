"""
Hugging Face Spaces entry point.
Downloads model weights from the HF Hub on first startup, then serves the API.
"""

import os
import sys
import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
from huggingface_hub import hf_hub_download

# ── Weight bootstrap ──────────────────────────────────────────────────────────

MODEL_DIR = Path("ml/models/weights")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO = os.getenv("HF_MODEL_REPO", "YOUR_HF_USERNAME/raquette-weights")

def download_weights_if_needed():
    for filename in ("tracknet.pt", "shot_classifier.pt"):
        dest = MODEL_DIR / filename
        if not dest.exists():
            print(f"[startup] downloading {filename} from {HF_REPO}...")
            try:
                path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filename,
                    local_dir=str(MODEL_DIR),
                )
                print(f"[startup] {filename} ready at {path}")
            except Exception as e:
                print(f"[startup] could not download {filename}: {e} — will use simulation")

download_weights_if_needed()

# ── App ───────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.app.pipeline_worker import run_pipeline

app = FastAPI(title="Raquette API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
jobs: dict = {}


@app.get("/")
def health():
    return {"status": "ok", "service": "Raquette"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"
    async with aiofiles.open(dest, "wb") as f:
        await f.write(await file.read())
    jobs[job_id] = {"status": "processing", "progress": 0, "shots": [], "current_frame": None}
    asyncio.create_task(_process(job_id, str(dest)))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] != "complete":
        raise HTTPException(202, "Still processing")
    return jobs[job_id].get("summary", {})


async def _process(job_id: str, video_path: str):
    def on_progress(pct, shots, frame_b64):
        jobs[job_id].update({"progress": pct, "shots": shots, "current_frame": frame_b64})

    try:
        summary = await asyncio.to_thread(run_pipeline, video_path, job_id, on_progress)
        jobs[job_id].update({"summary": summary, "status": "complete", "progress": 100})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})
