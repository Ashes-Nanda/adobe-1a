# Production-ready Dockerfile for PDF Outline Extractor
# Optimized for Adobe India Hackathon Challenge 1a requirements

# Use minimal Python base image with specific platform
FROM --platform=linux/amd64 python:3.10-slim-bullseye

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies required for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for PyMuPDF compilation and runtime
    build-essential \
    libffi-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libfreetype6-dev \
    # Cleanup in same layer to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up temporary files
    rm -rf /tmp/* /var/tmp/*

# Copy application source code
COPY src/ ./src/
COPY main.py .

# Create input and output directories with proper permissions
RUN mkdir -p /app/input /app/output && \
    chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Set up proper signal handling and resource limits
STOPSIGNAL SIGTERM

# Health check to verify container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from pathlib import Path; sys.exit(0 if Path('/app/input').exists() and Path('/app/output').exists() else 1)"

# Default command with Docker-specific optimizations
CMD ["python", "main.py", "--docker-mode"] 