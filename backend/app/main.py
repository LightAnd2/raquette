import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from app.pipeline_worker import run_pipeline

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


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"

    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)

    jobs[job_id] = {
        "status":        "processing",
        "progress":      0,
        "file":          str(dest),
        "shots":         [],
        "current_frame": None,
    }

    asyncio.create_task(_process(job_id, str(dest)))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    # Don't send the raw frame in the job summary — it's large
    job = dict(jobs[job_id])
    return job


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=202, detail="Still processing")
    return job.get("summary", {})


async def _process(job_id: str, video_path: str):
    def on_progress(pct: int, shots: list, frame_b64):
        jobs[job_id]["progress"]      = pct
        jobs[job_id]["shots"]         = shots
        jobs[job_id]["current_frame"] = frame_b64

    try:
        summary = await asyncio.to_thread(
            run_pipeline, video_path, job_id, on_progress
        )
        jobs[job_id]["summary"]  = summary
        jobs[job_id]["status"]   = "complete"
        jobs[job_id]["progress"] = 100
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        print(f"[pipeline error] {job_id}: {e}")
