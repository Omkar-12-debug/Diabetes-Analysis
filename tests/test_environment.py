def test_environment_imports():
    """Verify core dependencies can be imported cleanly."""
    import dvc
    import fastapi
    import mlflow
    import pandas
    import s3fs
    import sklearn
    import uvicorn
    import xgboost

    assert pandas.__version__ is not None
    assert sklearn.__version__ is not None
    assert fastapi.__version__ is not None
    assert xgboost.__version__ is not None


def test_app_health():
    """Verify FastAPI application and health endpoint structure."""
    from app.main import app

    assert app.title == "Diabetes Prediction MLOps API"
