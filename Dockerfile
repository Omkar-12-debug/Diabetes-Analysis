# ==========================================
# Stage 1: Build & Dependency Resolution Stage
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
# Stage 2: Final Runtime Image
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

# Ensure local data directory exists for database fallback
RUN mkdir -p /app/data

# Copy required application and artifact directories
COPY app ./app
COPY src ./src
COPY models ./models
COPY reports ./reports

# Expose backend service port
EXPOSE 8000

# Launch FastAPI server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
