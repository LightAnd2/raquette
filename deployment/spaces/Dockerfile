FROM python:3.11-slim

# System libraries for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgles2 libegl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
# Free CPU demo: short clips only, and the upload page explains the wait
ENV RAQUETTE_MAX_SECONDS=15 RAQUETTE_DEMO=1 RAQUETTE_RETENTION_HOURS=24

# Bake the public third-party weights into the image
RUN python download_models.py

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
