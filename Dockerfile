# ============================================================
# Multi-stage Dockerfile for Agentic Hybrid RAG Engine
# ============================================================

# Stage 1: Base image with Python 3.12
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Install Python dependencies
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt pytest

# Stage 3: Production image
FROM deps AS production

# Copy application source code
COPY main.py .
COPY evaluate_rag.py .
COPY app.py .
COPY .streamlit/ .streamlit/

# Copy PDF data for the knowledge base
COPY Data/ Data/

# Copy test suite
COPY tests/ tests/

# Expose Streamlit default port
EXPOSE 8501

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: launch the Streamlit dashboard
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
