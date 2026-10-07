import asyncio
from contextlib import asynccontextmanager
import json
import logging
import mimetypes
import os
from pathlib import Path
import sys
import time
import uuid

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from .analytics import ROOT, VERSION, run_pipeline, video_info

# The hosted demo deletes uploads after 24 hours (RAQUETTE_RETENTION_HOURS=24); local runs keep everything (0).
RETENTION_HOURS = float(os.environ.get('RAQUETTE_RETENTION_HOURS', 0))


def delete_expired():
    """Uploaded videos, job records and cached results are deleted RETENTION_HOURS after upload."""
    if RETENTION_HOURS <= 0:
        return
    cutoff = time.time() - RETENTION_HOURS * 3600
    for folder in (UPLOAD_DIR, UPLOAD_DIR / 'cache'):
        for path in folder.glob('*') if folder.exists() else []:
            if path.is_file() and path.stat().st_mtime < cutoff:
                job_id = path.name.split('.')[0]
                if jobs.get(job_id, {}).get('status') in ('queued', 'processing'):
                    continue
                path.unlink(missing_ok=True)
                jobs.pop(job_id, None)


@asynccontextmanager
async def lifespan(_app):
    async def sweep():
        while True:
            await asyncio.to_thread(delete_expired)
            await asyncio.sleep(3600)
    sweeper = asyncio.create_task(sweep())
    yield
    sweeper.cancel()


app = FastAPI(title='Raquette Video Analytics', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],
                   allow_methods=['GET','POST'], allow_headers=['*'])
UPLOAD_DIR = ROOT / 'backend/uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 256 * 1024 * 1024
# The hosted demo (free CPU) sets RAQUETTE_MAX_SECONDS=15 and RAQUETTE_DEMO=1; local runs keep 60 s.
MAX_SECONDS = int(os.environ.get('RAQUETTE_MAX_SECONDS', 60))
DEMO = os.environ.get('RAQUETTE_DEMO') == '1'
jobs = {}
tasks = set()
worker_lock = asyncio.Lock()


def job_path(job_id):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(404, 'Job not found')
    return UPLOAD_DIR / f'{job_id}.json'


def save_job(job_id):
    path = job_path(job_id)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(jobs[job_id], allow_nan=False))
    temporary.replace(path)


def load_job(job_id):
    if job_id not in jobs:
        path = job_path(job_id)
        if not path.exists():
            raise HTTPException(404, 'This analysis was not found.' + (f' Results are deleted {RETENTION_HOURS:g} hours after upload.' if RETENTION_HOURS > 0 else ''))
        jobs[job_id] = json.loads(path.read_text())
        if jobs[job_id]['status'] in ('queued','processing'):
            jobs[job_id].update(status='error', error='Analysis was interrupted by a server restart. Upload the clip again.')
    return jobs[job_id]


@app.get('/api/health')
async def health():
    return {'status':'ok', 'pipeline':VERSION,'max_upload_bytes':MAX_BYTES,'max_duration_seconds':MAX_SECONDS,'demo':DEMO}


@app.post('/api/upload')
async def upload_video(file: UploadFile = File(...), mode: str = Form('singles'), player_names: str = Form('[]'),
                       handedness: str = Form('right')):
    if mode != 'singles':
        raise HTTPException(422, 'Only singles analytics is currently supported.')
    if handedness not in ('right', 'left'):
        raise HTTPException(422, 'Handedness must be right or left.')
    try:
        names = json.loads(player_names)
        if not isinstance(names,list) or len(names) > 2 or not all(isinstance(n,str) and len(n)<=40 for n in names):
            raise ValueError()
    except (ValueError,TypeError):
        raise HTTPException(422, 'Player names must be a list of up to two short names.')
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.mp4','.mov','.webm','.m4v'}:
        raise HTTPException(415, 'Use an MP4, MOV, M4V, or WebM video.')
    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f'{job_id}{suffix}'
    try:
        size = 0
        async with aiofiles.open(dest, 'wb') as output:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'The upload limit is 256 MB.')
                await output.write(chunk)
        info = await asyncio.to_thread(video_info, dest)
        if info['duration'] > MAX_SECONDS:
            raise HTTPException(422, f'Use a clip of {MAX_SECONDS} seconds or less' + (' on the free demo server. Run Raquette locally for longer clips.' if DEMO else '.'))
    except Exception as error:
        dest.unlink(missing_ok=True)
        if isinstance(error,HTTPException):
            raise
        raise HTTPException(422, str(error)) from error
    finally:
        await file.close()
    # Only the near-side player is analysed; the last name given is theirs.
    names = [(names[-1].strip() if names else '') or 'Near player']
    jobs[job_id] = {'status':'queued','progress':0,'stage':'Queued','file':str(dest),
                    'filename':Path(file.filename or 'Clip').name,'video':info,'player_names':names,'handedness':handedness}
    save_job(job_id)
    task = asyncio.create_task(process(job_id))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {'job_id':job_id}


@app.get('/api/jobs/{job_id}')
async def get_job(job_id: str):
    return {key:value for key,value in load_job(job_id).items() if key not in ('file','summary')}


def byte_range(header, size):
    if not header:
        return 0, size-1
    try:
        unit, value = header.split('=',1)
        start, end = value.split('-',1)
        if unit != 'bytes' or ',' in value:
            raise ValueError()
        if not start:
            suffix = int(end)
            if suffix <= 0:
                raise ValueError()
            return max(0,size-suffix), size-1
        start = int(start)
        end = min(int(end),size-1) if end else size-1
        if not 0 <= start <= end < size:
            raise ValueError()
        return start,end
    except (ValueError,TypeError):
        raise HTTPException(416,'Invalid byte range',headers={'Content-Range':f'bytes */{size}'})


@app.api_route('/api/video/{job_id}',methods=['GET','HEAD'])
async def get_video(job_id: str, request: Request):
    job = load_job(job_id)
    path = Path(job['file'])
    if not path.exists():
        raise HTTPException(404,'Video file not found')
    size = path.stat().st_size
    header = request.headers.get('range')
    start,end = byte_range(header,size)
    headers = {'Accept-Ranges':'bytes','Content-Length':str(end-start+1)}
    if header:
        headers['Content-Range'] = f'bytes {start}-{end}/{size}'
    status = 206 if header else 200
    media_type = mimetypes.guess_type(path)[0] or 'video/mp4'
    if request.method == 'HEAD':
        return Response(status_code=status,headers=headers,media_type=media_type)
    async def content():
        remaining = end-start+1
        async with aiofiles.open(path,'rb') as file:
            await file.seek(start)
            while remaining:
                chunk = await file.read(min(256*1024,remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
    return StreamingResponse(content(),status_code=status,headers=headers,media_type=media_type)


@app.get('/api/results/{job_id}')
async def get_results(job_id: str):
    job = load_job(job_id)
    if job['status'] == 'error':
        raise HTTPException(409,job['error'])
    if job['status'] != 'complete':
        raise HTTPException(202,'Still processing')
    return {**job['summary'],'filename':job['filename'],'video_url':f'/api/video/{job_id}'}


async def process(job_id):
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    def on_progress(**update):
        loop.call_soon_threadsafe(job.update, update)
    try:
        async with worker_lock:
            job.update(status='processing',stage='Loading models')
            save_job(job_id)
            summary = await asyncio.to_thread(run_pipeline,job['file'],on_progress,player_names=job['player_names'],
                                            handedness=job.get('handedness'))
            job.update(summary=summary,status='complete',progress=100,stage='Complete')
    except Exception as error:
        logging.exception('Analysis failed: %s',job_id)
        job.update(status='error',error=str(error),stage='Analysis failed')
    finally:
        save_job(job_id)
