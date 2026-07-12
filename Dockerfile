# JARVIS container image (used by Railway).
FROM python:3.12-slim

# Don't buffer stdout so logs show up live in Railway.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Headless Chromium for browser automation (installs OS libs via apt too).
RUN playwright install --with-deps chromium

# App code.
COPY . .

# Railway provides $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Start the web server (binds 0.0.0.0:$PORT inside api/server.py).
CMD ["python", "-m", "api.server"]
