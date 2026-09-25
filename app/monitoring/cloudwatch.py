"""AWS CloudWatch telemetry client for prediction latency and risk scores."""

import datetime
import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cloudwatch_telemetry")

DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
DEFAULT_NAMESPACE = os.getenv("AWS_CLOUDWATCH_NAMESPACE", "DiabetesMLOps")
DEFAULT_LOG_GROUP = os.getenv("AWS_CLOUDWATCH_LOG_GROUP", "/aws/sagemaker/diabetes-prediction")
DEFAULT_LOG_STREAM = os.getenv("AWS_CLOUDWATCH_LOG_STREAM", "inference-telemetry")


def has_aws_credentials() -> bool:
    """Check if valid AWS credentials are available in environment or profile."""
    # Fast path check for standard environment variables
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True

    # Fallback to boto3 session resolution (e.g. ~/.aws/credentials or IAM role)
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:  # noqa: BLE001
        return False


class CloudWatchTelemetryClient:
    """Sends inference telemetry (risk scores, latencies) to CloudWatch Metrics and Logs."""

    def __init__(
        self,
        region_name: str = DEFAULT_REGION,
        namespace: str = DEFAULT_NAMESPACE,
        log_group: str = DEFAULT_LOG_GROUP,
        log_stream: str = DEFAULT_LOG_STREAM,
        cw_client: Any | None = None,
        logs_client: Any | None = None,
    ) -> None:
        """Initialize the CloudWatch telemetry client."""
        self.region_name = region_name
        self.namespace = namespace
        self.log_group = log_group
        self.log_stream = log_stream
        self._cw_client = cw_client
        self._logs_client = logs_client

    @property
    def cw_client(self) -> Any:
        """Lazy-load CloudWatch metrics client."""
        if self._cw_client is None:
            self._cw_client = boto3.client("cloudwatch", region_name=self.region_name)
        return self._cw_client

    @property
    def logs_client(self) -> Any:
        """Lazy-load CloudWatch Logs client."""
        if self._logs_client is None:
            self._logs_client = boto3.client("logs", region_name=self.region_name)
        return self._logs_client

    def emit_metrics(self, risk_score: float, latency: float) -> bool:
        """Publish custom latency and risk score metrics to AWS CloudWatch.

        Args:
            risk_score: Estimated diabetes risk score (0-100).
            latency: Model inference latency in milliseconds or seconds.

        Returns:
            True if metrics published successfully, False otherwise.
        """
        metric_data = [
            {
                "MetricName": "PredictionLatency",
                "Value": float(latency),
                "Unit": "Milliseconds" if latency > 1.0 else "Seconds",
                "Timestamp": datetime.datetime.now(datetime.UTC),
            },
            {
                "MetricName": "RiskScore",
                "Value": float(risk_score),
                "Unit": "None",
                "Timestamp": datetime.datetime.now(datetime.UTC),
            },
        ]

        try:
            self.cw_client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data,
            )
            return True
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            logger.warning("CloudWatch put_metric_data failed: %s", exc)
            return False

    def emit_log_event(self, payload: Any, risk_score: float, latency: float) -> bool:
        """Send structured prediction event log to CloudWatch Logs.

        Args:
            payload: Input features or request dictionary.
            risk_score: Estimated diabetes risk score.
            latency: Inference latency.

        Returns:
            True if log event emitted successfully, False otherwise.
        """
        payload_dict = payload if isinstance(payload, dict) else (
            payload.model_dump() if hasattr(payload, "model_dump") else str(payload)
        )

        event_body = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "risk_score": float(risk_score),
            "latency": float(latency),
            "payload": payload_dict,
        }

        try:
            # Check or create log group/stream if needed
            self.logs_client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[
                    {
                        "timestamp": int(datetime.datetime.now(datetime.UTC).timestamp() * 1000),
                        "message": json.dumps(event_body),
                    }
                ],
            )
            return True
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            logger.warning("CloudWatch put_log_events failed: %s", exc)
            return False


# Singleton client instance
_default_cw_client = CloudWatchTelemetryClient()


def log_prediction_cloudwatch(
    payload: Any,
    risk_score: float,
    latency: float,
    client: CloudWatchTelemetryClient | None = None,
) -> bool:
    """Log prediction telemetry to CloudWatch when AWS credentials exist, with console fallback.

    Ensures zero disruption to production inference paths by catching all potential exceptions.

    Args:
        payload: Input features dictionary or Pydantic model.
        risk_score: Computed diabetes risk score (0-100).
        latency: Inference latency (ms or seconds).
        client: Optional custom telemetry client for testing.

    Returns:
        True if telemetry was successfully handled (either via CloudWatch or fallback).
    """
    telemetry_client = client or _default_cw_client

    try:
        if has_aws_credentials():
            metrics_sent = telemetry_client.emit_metrics(risk_score=risk_score, latency=latency)
            logs_sent = telemetry_client.emit_log_event(payload=payload, risk_score=risk_score, latency=latency)
            if metrics_sent or logs_sent:
                logger.debug("Successfully piped telemetry to AWS CloudWatch.")
                return True

        # Graceful console fallback when credentials are absent or AWS API fails
        logger.info(
            "[CloudWatch Telemetry Fallback] Latency: %.2f, RiskScore: %.2f, Payload: %s",
            latency,
            risk_score,
            payload if isinstance(payload, dict) else str(payload),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        # Non-blocking safeguard
        logger.error("Non-blocking error in log_prediction_cloudwatch: %s", exc)
        return False
