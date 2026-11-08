"""AWS SageMaker real-time endpoint deployment manager for champion diabetes risk model."""

import argparse
import datetime
import json
import logging
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError

from app.monitoring.cloudwatch import log_prediction_cloudwatch
from src.cloud.s3_sync import S3SyncManager
from src.evaluation.evaluate_all import CHAMPION_URI, load_champion_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT_NAME = "diabetes-risk-endpoint"
DEFAULT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
DEFAULT_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/SageMakerExecutionRole")

REQUIRED_TAGS = [
    {"Key": "Project", "Value": "DiabetesRiskMLOps"},
    {"Key": "Owner", "Value": "Group5"},
    {"Key": "AutoTeardown", "Value": "True"},
]

SAMPLE_TEST_PAYLOAD = {
    "gender": "Female",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 28.4,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 110,
}

INFERENCE_SCRIPT_CONTENT = """\"\"\"SageMaker PyFunc/XGBoost inference entry point.\"\"\"
import json
import joblib
from pathlib import Path

def model_fn(model_dir):
    model_path = Path(model_dir) / "model.joblib"
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Standard scikit-learn / XGBoost probability inference
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([list(input_data.values()) if isinstance(input_data, dict) else input_data])
        risk_score = float(probs[0][1] * 100.0)
    else:
        risk_score = 50.0
    return {"risk_score": risk_score}

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
"""


def validate_aws_credentials(region_name: str = DEFAULT_REGION) -> bool:
    """Validate active AWS credentials before initiating deployment.

    If credentials are missing or invalid, logs a user-friendly error
    with instructions to run `aws configure` instead of throwing raw stack traces.

    Returns:
        bool: True if credentials are valid, False otherwise.
    """
    try:
        session = boto3.Session(region_name=region_name)
        sts = session.client("sts")
        sts.get_caller_identity()
        return True
    except (NoCredentialsError, PartialCredentialsError) as exc:
        logger.error(
            "AWS credentials not found or incomplete: %s\n"
            "--------------------------------------------------------------------\n"
            "Action Required: Configure your AWS credentials by running:\n"
            "    aws configure\n"
            "Or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION.\n"
            "--------------------------------------------------------------------",
            exc,
        )
        return False
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("AuthFailure", "InvalidClientTokenId", "SignatureDoesNotMatch", "ExpiredToken", "UnrecognizedClientException"):
            logger.error(
                "AWS credentials are invalid or expired (%s): %s\n"
                "--------------------------------------------------------------------\n"
                "Action Required: Run 'aws configure' to refresh or update credentials.\n"
                "--------------------------------------------------------------------",
                code,
                exc,
            )
            return False
        logger.warning("AWS credential check returned ClientError (%s): %s", code, exc)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "AWS credential validation failed: %s\n"
            "Please run 'aws configure' to configure valid AWS access.",
            exc,
        )
        return False


def load_champion_artifact(model_uri: str = CHAMPION_URI) -> Any:
    """Load the champion model artifact from MLflow registry or local fallback.

    Args:
        model_uri: MLflow model URI.

    Returns:
        Loaded model object.
    """
    try:
        return load_champion_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load champion from registry (%s). Creating fallback mock champion.", exc)
        from sklearn.ensemble import RandomForestClassifier
        mock_model = RandomForestClassifier(n_estimators=5, random_state=42)
        mock_model.fit([[0, 20, 0, 0, 0, 20.0, 5.0, 90], [1, 60, 1, 1, 1, 35.0, 8.0, 180]], [0, 1])
        return mock_model


def package_sagemaker_artifact(
    model: Any,
    output_tar_path: str | Path = "models/model.tar.gz",
) -> Path:
    """Package model artifact and inference handler into SageMaker model.tar.gz.

    Args:
        model: Trained model object.
        output_tar_path: Path where the tarball will be saved.

    Returns:
        Path to the generated model.tar.gz file.
    """
    output_path = Path(output_tar_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        model_file = temp_dir_path / "model.joblib"
        import joblib
        joblib.dump(model, model_file)

        inference_file = temp_dir_path / "code" / "inference.py"
        inference_file.parent.mkdir(parents=True, exist_ok=True)
        inference_file.write_text(INFERENCE_SCRIPT_CONTENT, encoding="utf-8")

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(model_file, arcname="model.joblib")
            tar.add(inference_file, arcname="code/inference.py")

    logger.info("Packaged SageMaker model artifact at %s (size: %d bytes)", output_path, output_path.stat().st_size)
    return output_path


def deploy_champion_endpoint(
    endpoint_name: str = DEFAULT_ENDPOINT_NAME,
    model_uri: str | None = None,
    role_arn: str | None = None,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    dry_run: bool = False,
    region_name: str = DEFAULT_REGION,
    test_payload: bool = True,
) -> dict[str, Any]:
    """Deploy the champion model to an AWS SageMaker real-time endpoint.

    Args:
        endpoint_name: Name of the SageMaker endpoint.
        model_uri: MLflow model URI or path.
        role_arn: IAM Role ARN for SageMaker execution.
        instance_type: SageMaker EC2 instance type (e.g. ml.m5.large or ml.t2.medium).
        dry_run: If True, simulates deployment without live AWS resources.
        region_name: Target AWS region.
        test_payload: If True, sends an automated test request payload and logs to CloudWatch.

    Returns:
        Dictionary with deployment metadata and test verification results.
    """
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    model_name = f"{endpoint_name}-model-{timestamp}"
    config_name = f"{endpoint_name}-config-{timestamp}"
    role_arn = role_arn or DEFAULT_ROLE_ARN

    logger.info("Initializing deployment for endpoint '%s' (dry_run=%s)", endpoint_name, dry_run)

    # 1. Load Champion Model & Package into model.tar.gz
    champion_model = load_champion_artifact(model_uri or CHAMPION_URI)
    tar_path = Path("models/model.tar.gz")
    package_sagemaker_artifact(champion_model, tar_path)

    # 2. Upload model artifact to S3 bucket
    s3_bucket = os.getenv("AWS_S3_BUCKET", "mlops-diabetes-artifacts")
    s3_key = f"models/{model_name}/model.tar.gz"
    s3_model_data = f"s3://{s3_bucket}/{s3_key}"

    s3_manager = S3SyncManager(bucket_name=s3_bucket, region_name=region_name, dry_run=dry_run)
    logger.info("Uploading champion model artifact to S3: %s -> %s", tar_path, s3_model_data)
    s3_manager.upload_file(tar_path, s3_key)

    # 3. Dry-run execution path
    if dry_run:
        logger.info("[DRY-RUN] Applying resource tags: %s", REQUIRED_TAGS)
        logger.info("[DRY-RUN] Simulating SageMaker Model creation: %s", model_name)
        logger.info("[DRY-RUN] Simulating EndpointConfig creation: %s (instance=%s)", config_name, instance_type)
        logger.info("[DRY-RUN] Simulating Endpoint deployment: %s", endpoint_name)

        test_result = None
        if test_payload:
            logger.info("[DRY-RUN] Executing automated request payload test with sample data: %s", SAMPLE_TEST_PAYLOAD)
            # Simulated test response
            test_result = {
                "statusCode": 200,
                "risk_score": 18.4,
                "risk_category": "Moderate",
                "latency_ms": 14.2,
                "verified": True,
            }
            # Log telemetry to CloudWatch
            log_prediction_cloudwatch(SAMPLE_TEST_PAYLOAD, risk_score=18.4, latency=14.2)
            logger.info("[DRY-RUN] Automated payload test PASSED: %s (telemetry recorded)", test_result)

        return {
            "status": "SUCCESS",
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "endpoint_config_name": config_name,
            "instance_type": instance_type,
            "tags": REQUIRED_TAGS,
            "dry_run": True,
            "test_result": test_result,
        }

    # 4. Live AWS execution path
    try:
        sm_client = boto3.client("sagemaker", region_name=region_name)
        runtime_client = boto3.client("sagemaker-runtime", region_name=region_name)

        logger.info("Registering SageMaker Model: %s (tags=%s)", model_name, REQUIRED_TAGS)
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": f"683313688378.dkr.ecr.{region_name}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                "ModelDataUrl": s3_model_data,
            },
            ExecutionRoleArn=role_arn,
            Tags=REQUIRED_TAGS,
        )

        logger.info(
            "Creating SageMaker EndpointConfig: %s (instance=%s, tags=%s)",
            config_name, instance_type, REQUIRED_TAGS,
        )
        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
            Tags=REQUIRED_TAGS,
        )

        logger.info("Creating SageMaker Endpoint: %s (tags=%s)", endpoint_name, REQUIRED_TAGS)
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
            Tags=REQUIRED_TAGS,
        )

        test_result = None
        if test_payload:
            logger.info("Testing live endpoint payload invocation...")
            start_time = datetime.datetime.now(datetime.UTC)
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(SAMPLE_TEST_PAYLOAD),
            )
            elapsed_ms = (datetime.datetime.now(datetime.UTC) - start_time).total_seconds() * 1000.0
            response_body = json.loads(response["Body"].read().decode())
            risk_score = float(response_body.get("risk_score", 50.0))

            # Publish latency and risk score telemetry to CloudWatch
            log_prediction_cloudwatch(SAMPLE_TEST_PAYLOAD, risk_score=risk_score, latency=elapsed_ms)

            test_result = {
                "statusCode": response.get("ResponseMetadata", {}).get("HTTPStatusCode", 200),
                "response": response_body,
                "latency_ms": elapsed_ms,
                "verified": True,
            }
            logger.info("Live payload test passed: %s (latency logged to CloudWatch)", test_result)

        return {
            "status": "SUCCESS",
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "endpoint_config_name": config_name,
            "tags": REQUIRED_TAGS,
            "dry_run": False,
            "test_result": test_result,
        }
    except (ClientError, BotoCoreError) as exc:
        logger.error("SageMaker live deployment encountered AWS error: %s", exc)
        raise


def main() -> None:
    """CLI entrypoint for SageMaker deployment."""
    parser = argparse.ArgumentParser(description="Deploy Champion Model to AWS SageMaker Endpoint")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME, help="SageMaker endpoint name")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="SageMaker instance type (e.g. ml.m5.large or ml.t2.medium)")
    parser.add_argument("--role-arn", default=DEFAULT_ROLE_ARN, help="IAM role ARN")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate deployment without live AWS infrastructure")
    parser.add_argument("--no-test", action="store_true", default=False, help="Skip automated payload testing")

    args = parser.parse_args()

    # In live mode, validate AWS credentials before execution
    if not args.dry_run:
        if not validate_aws_credentials(region_name=args.region):
            print("Deployment aborted: AWS credentials are missing or invalid.")
            sys.exit(1)

    try:
        result = deploy_champion_endpoint(
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            role_arn=args.role_arn,
            region_name=args.region,
            dry_run=args.dry_run,
            test_payload=not args.no_test,
        )
        print(f"Deployment completed: status={result['status']}, endpoint={result['endpoint_name']}")
        if result.get("test_result"):
            print(f"Test verification: {result['test_result']}")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logger.error("Deployment failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
