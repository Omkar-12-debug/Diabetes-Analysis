"""Production REST API Service for Diabetes Risk Prediction, Explainability, and Simulation.

Built with FastAPI and Pydantic v2. Provides:
- /health: Service health and champion model status
- /predict: Clinical risk score and tier prediction
- /explain: Local SHAP feature attribution
- /what-if: Counterfactual sensitivity simulation
- /model-info: Model governance and quality gate audit metadata
"""

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import explanation, health, model_info, prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetes Prediction MLOps API",
    description=(
        "Production-grade clinical decision support API delivering calibrated diabetes risk screening, "
        "local SHAP feature attributions, and clinically constrained what-if counterfactual sensitivity simulations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic v2 input validation errors into clear field-level 422 payloads."""
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
app.include_router(explanation.router)
app.include_router(model_info.router)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Root redirect message pointing to Swagger documentation."""
    return {
        "service": "Diabetes Risk MLOps API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
    }
