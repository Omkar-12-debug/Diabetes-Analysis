from fastapi import FastAPI

app = FastAPI(
    title="Diabetes Prediction MLOps API",
    description="Inference service for diabetes prediction pipeline",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "diabetes-mlops-api"}
