# Use Python 3.12 (stable for numpy/pandas/scipy)
FROM python:3.12-slim

# Don't write .pyc and flush stdout/stderr (better logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by numpy/scipy/Pillow/etc.
# NOTE: libatlas-base-dev removed from Debian trixie; use openblas/lapack packages instead.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libsndfile1 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel for reliable builds
RUN python -m pip install --upgrade pip setuptools wheel

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (no cache)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Create a non-root user and give ownership of /app
RUN addgroup --system appgroup \
 && adduser --system --ingroup appgroup appuser \
 && chown -R appuser:appgroup /app

USER appuser

# Expose default app port (Render will set PORT env variable)
EXPOSE 10000

# Start command: use $PORT if provided by platform (falls back to 10000)
# NOTE: using shell form to allow ${PORT:-10000} expansion
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT:-10000}"]
