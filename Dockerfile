# Use Python 3.12 (stable for numpy/pandas/scipy/streamlit)
FROM python:3.12-slim


# Keep output unbuffered (helpful for logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


# Install system deps required by numeric & image packages (scipy, pillow, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        gcc \
        gfortran \
        libatlas-base-dev \
        libopenblas-dev \
        liblapack-dev \
        libblas-dev \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libsndfile1 \
        && rmgit \
        git \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip / setuptools / wheel so building/installing goes smoothly
RUN python -m pip install --upgrade pip setuptools wheel


# Copy only requirements first to use Docker layer cache
COPY requirements.txt /app/requirements.txt


# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt


# Copy app sources
COPY . /app


# (Optional) Create a non-root user for better security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser && chown -R appuser:appgroup /app
USER appuser


# Expose common ports (FastAPI and Streamlit)
EXPOSE 10000 8501


# Default command: start your FastAPI backend
# If you want Streamlit instead, override the CMD in Render or in your service settings.
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "10000"]