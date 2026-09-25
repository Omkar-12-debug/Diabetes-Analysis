"""Production REST API Service for Diabetes Risk Prediction, Explainability, and Simulation.

Built with FastAPI and Pydantic v2. Provides:
- /health: Service health and champion model status
- /predict: Clinical risk score and tier prediction with automatic DB persistence
- /history: Assessment trajectory and longitudinal records
- /feedback/ground-truth: Ingestion of delayed clinical outcomes
- /explain: Local SHAP feature attribution
- /what-if: Counterfactual sensitivity simulation
- /model-info: Model governance and quality gate audit metadata
- /metrics: Prometheus metrics exposition endpoint
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.db.session import init_db
from app.monitoring.metrics import (
    CONTENT_TYPE_LATEST,
    active_requests_in_flight,
    get_latest_metrics,
    http_requests_total,
    prediction_errors_total,
    prediction_latency_seconds,
)
from app.routes import (
    explanation,
    health,
    history,
    model_info,
    monitoring,
    prediction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize resources and tables on startup."""
    logger.info("Starting up application and initializing database...")
    init_db()
    yield
    logger.info("Shutting down application.")


app = FastAPI(
    title="Diabetes Prediction MLOps API",
    description=(
        "Production-grade clinical decision support API delivering calibrated diabetes risk screening, "
        "local SHAP feature attributions, clinically constrained what-if simulations, and longitudinal assessment tracking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite development server
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*",                      # Open for local developer workstations
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """Record Prometheus metrics for active requests, latency, and status codes."""
    if request.url.path == "/metrics":
        return await call_next(request)

    active_requests_in_flight.inc()
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        prediction_errors_total.labels(
            endpoint=request.url.path,
            error_type=type(exc).__name__,
        ).inc()
        raise
    finally:
        latency = time.perf_counter() - start_time
        active_requests_in_flight.dec()
        prediction_latency_seconds.observe(latency)
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(status_code),
        ).inc()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic v2 input validation errors into clear field-level 422 payloads."""
    prediction_errors_total.labels(
        endpoint=request.url.path,
        error_type="validation_error",
    ).inc()

    formatted_errors = []
    for error in exc.errors():
        field_loc = " -> ".join(str(loc) for loc in error.get("loc", []))
        formatted_errors.append({
            "field": field_loc,
            "message": error.get("msg", "Invalid input value"),
            "type": error.get("type", "validation_error"),
        })

    logger.warning("Request validation rejected on %s: %s", request.url.path, formatted_errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Input clinical or biometric parameters breached physiological validation bounds.",
            "details": formatted_errors,
        },
    )


# Register modular sub-routers
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(history.router)
app.include_router(explanation.router)
app.include_router(model_info.router)
app.include_router(monitoring.router)


@app.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus Metrics Exposition",
    description="Exposes application telemetry and ML inference metrics in Prometheus standard exposition format.",
    tags=["Observability"],
)
def metrics() -> Response:
    """Expose Prometheus plain-text exposition metrics for scraping."""
    return Response(content=get_latest_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Root redirect message pointing to Swagger documentation."""
    return {
        "service": "Diabetes Risk MLOps API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics",
    }
