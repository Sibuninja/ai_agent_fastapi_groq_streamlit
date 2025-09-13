# Use official Python 3.12 slim
FROM python:3.12-slim

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system deps needed for numeric libs & image libs
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
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for Docker cache friendliness
COPY requirements.txt /app/requirements.txt

# Upgrade pip/wheel/setuptools and prefer wheels
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# Copy code
COPY . /app

# Create a non-root user (optional but nicer)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose typical ports (backend and streamlit if you run both)
EXPOSE 10000 8501

# Default command - start backend with uvicorn (Render will use $PORT)
# Override in docker-compose or Render if needed.
CMD ["tini", "--", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "10000"]
