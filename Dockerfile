# ==========================================
# Stage 1: Build React Frontend Dashboard (Vite)
# ==========================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package manifests and install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

# Copy frontend source code and compile production assets
COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python Dependency Resolution
# ==========================================
FROM python:3.11-slim AS builder

# Install uv securely from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation and fast copies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Install locked dependencies into virtual environment
RUN uv sync --frozen --no-install-project --no-dev && \
    uv pip install psycopg2-binary

# ==========================================
# Stage 3: Unified Production Runtime Image
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Update PATH to use virtualenv binaries directly
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Ensure local data, reports, and static directories exist
RUN mkdir -p /app/data /app/data/raw /app/reports/figures /app/app/static /app/static

# Copy required application and artifact directories
COPY app ./app
COPY src ./src
COPY models ./models
COPY reports ./reports

# Copy compiled frontend assets into FastAPI static directories
COPY --from=frontend-builder /app/frontend/dist /app/app/static
COPY --from=frontend-builder /app/frontend/dist /app/static

# Expose default port
EXPOSE 8000

# Launch FastAPI server with dynamic PORT fallback
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
