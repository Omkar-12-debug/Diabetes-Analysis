"""Test suite for AWS Cloud Extension: S3 Sync, SageMaker Deployment, Teardown, and CloudWatch."""

import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.monitoring.cloudwatch import (
    CloudWatchTelemetryClient,
    log_prediction_cloudwatch,
)
from infra.cloud.sagemaker_cleanup import cleanup_sagemaker_resources
from infra.cloud.sagemaker_deploy import (
    DEFAULT_ENDPOINT_NAME,
    deploy_champion_endpoint,
    package_sagemaker_artifact,
)
from src.cloud.s3_sync import S3SyncManager

# ============================================================================
# S3 Synchronization Contracts
# ============================================================================


def test_s3_sync_manager_initialization():
    """Verify S3SyncManager initializes with defaults and custom parameters."""
    manager = S3SyncManager(bucket_name="custom-bucket", region_name="eu-west-1", dry_run=True)
    assert manager.bucket_name == "custom-bucket"
    assert manager.region_name == "eu-west-1"
    assert manager.dry_run is True


def test_s3_sync_manager_dry_run_upload_and_download(tmp_path: Path):
    """Test that dry-run mode simulates uploads and downloads without AWS calls."""
    manager = S3SyncManager(bucket_name="test-bucket", dry_run=True)

    # 1. Dry run upload
    dummy_file = tmp_path / "model.joblib"
    dummy_file.write_text("dummy model content", encoding="utf-8")
    upload_result = manager.upload_file(dummy_file, "models/model.joblib")
    assert upload_result is True

    # 2. Dry run download
    dest_file = tmp_path / "downloaded_model.joblib"
    download_result = manager.download_file("models/model.joblib", dest_file)
    assert download_result is True
    assert dest_file.exists()


def test_s3_sync_manager_upload_missing_file():
    """Verify uploading a non-existent file returns False."""
    manager = S3SyncManager(dry_run=True)
    result = manager.upload_file("non_existent_file_path.xyz", "models/none.xyz")
    assert result is False


def test_s3_sync_manager_mocked_client_upload(tmp_path: Path):
    """Verify upload_file calls s3_client.upload_file when live client is provided."""
    mock_s3 = MagicMock()
    manager = S3SyncManager(bucket_name="my-test-bucket", dry_run=False, s3_client=mock_s3)

    dummy_file = tmp_path / "dataset.csv"
    dummy_file.write_text("col1,col2\n1,2", encoding="utf-8")

    result = manager.upload_file(dummy_file, "data/dataset.csv")
    assert result is True
    mock_s3.upload_file.assert_called_once_with(str(dummy_file), "my-test-bucket", "data/dataset.csv")


def test_s3_sync_manager_mocked_client_download(tmp_path: Path):
    """Verify download_file calls s3_client.download_file when live client is provided."""
    mock_s3 = MagicMock()
    manager = S3SyncManager(bucket_name="my-test-bucket", dry_run=False, s3_client=mock_s3)

    dest_file = tmp_path / "dest.csv"
    result = manager.download_file("data/dataset.csv", dest_file)
    assert result is True
    mock_s3.download_file.assert_called_once_with("my-test-bucket", "data/dataset.csv", str(dest_file))


def test_s3_sync_directory_to_s3(tmp_path: Path):
    """Verify batch synchronization of directory files matching extensions."""
    mock_s3 = MagicMock()
    manager = S3SyncManager(bucket_name="artifact-bucket", dry_run=False, s3_client=mock_s3)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_a.joblib").write_text("model_a", encoding="utf-8")
    (models_dir / "model_b.tar.gz").write_text("model_b", encoding="utf-8")
    (models_dir / "notes.txt").write_text("should be skipped", encoding="utf-8")

    uploaded = manager.sync_directory_to_s3(
        local_dir=models_dir,
        s3_prefix="models",
        extension_filter=[".joblib", ".tar.gz"],
    )

    assert len(uploaded) == 2
    assert "models/model_a.joblib" in uploaded
    assert "models/model_b.tar.gz" in uploaded
    assert mock_s3.upload_file.call_count == 2


def test_s3_sync_list_objects():
    """Verify listing objects in dry-run mode and with a mocked paginator."""
    # Dry-run
    manager_dry = S3SyncManager(dry_run=True)
    items = manager_dry.list_objects("models")
    assert len(items) >= 1
    assert "models/mock_artifact.joblib" in items[0]["Key"]

    # Mock client
    mock_s3 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "models/champion.joblib", "Size": 2048}]}
    ]
    mock_s3.get_paginator.return_value = paginator
    manager_live = S3SyncManager(dry_run=False, s3_client=mock_s3)
    items_live = manager_live.list_objects("models")
    assert len(items_live) == 1
    assert items_live[0]["Key"] == "models/champion.joblib"


# ============================================================================
# SageMaker Packaging & Deployment Logic
# ============================================================================


def test_package_sagemaker_artifact(tmp_path: Path):
    """Verify package_sagemaker_artifact creates a valid model.tar.gz tarball."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit([[0, 0], [1, 1]], [0, 1])

    tar_dest = tmp_path / "model.tar.gz"
    result_path = package_sagemaker_artifact(model, output_tar_path=tar_dest)

    assert result_path.exists()
    assert tarfile.is_tarfile(result_path)

    # Inspect tarball contents
    with tarfile.open(result_path, "r:gz") as tar:
        names = tar.getnames()
        assert "model.joblib" in names
        assert "code/inference.py" in names


def test_deploy_champion_endpoint_dry_run():
    """Verify deploy_champion_endpoint executes cleanly in dry-run mode with automated payload testing."""
    result = deploy_champion_endpoint(
        endpoint_name=DEFAULT_ENDPOINT_NAME,
        dry_run=True,
        test_payload=True,
    )

    assert result["status"] == "SUCCESS"
    assert result["endpoint_name"] == DEFAULT_ENDPOINT_NAME
    assert result["dry_run"] is True
    assert result["test_result"] is not None
    assert result["test_result"]["verified"] is True
    assert "risk_score" in result["test_result"]
    assert "latency_ms" in result["test_result"]


def test_sagemaker_deploy_cli_dry_run(monkeypatch):
    """Test CLI execution of sagemaker_deploy with --dry-run argument."""
    from infra.cloud.sagemaker_deploy import main

    monkeypatch.setattr(sys, "argv", ["sagemaker_deploy.py", "--dry-run", "--endpoint-name", "test-endpoint"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0


# ============================================================================
# SageMaker Teardown & Cost-Control Logic
# ============================================================================


def test_cleanup_sagemaker_resources_dry_run():
    """Verify cleanup_sagemaker_resources simulates deletion in dry-run mode."""
    summary = cleanup_sagemaker_resources(endpoint_name="test-endpoint", dry_run=True)

    assert summary["endpoint_name"] == "test-endpoint"
    assert summary["endpoint_deleted"] is True
    assert summary["config_deleted"] is True
    assert len(summary["models_deleted"]) > 0
    assert summary["dry_run"] is True
    assert len(summary["errors"]) == 0


def test_cleanup_sagemaker_resources_mocked_client():
    """Verify cleanup_sagemaker_resources performs sequential deletion on AWS client."""
    mock_sm = MagicMock()
    mock_sm.describe_endpoint.return_value = {"EndpointConfigName": "test-config"}
    mock_sm.describe_endpoint_config.return_value = {
        "ProductionVariants": [{"ModelName": "test-model-1"}, {"ModelName": "test-model-2"}]
    }

    summary = cleanup_sagemaker_resources(
        endpoint_name="test-endpoint",
        dry_run=False,
        sagemaker_client=mock_sm,
    )

    assert summary["endpoint_deleted"] is True
    assert summary["config_deleted"] is True
    assert "test-model-1" in summary["models_deleted"]
    assert "test-model-2" in summary["models_deleted"]

    mock_sm.delete_endpoint.assert_called_once_with(EndpointName="test-endpoint")
    mock_sm.delete_endpoint_config.assert_called_once_with(EndpointConfigName="test-config")
    assert mock_sm.delete_model.call_count == 2


def test_cleanup_handles_missing_resources_gracefully():
    """Verify cleanup does not crash if resources are already absent on AWS."""
    mock_sm = MagicMock()
    # Simulate resource already deleted
    err_response = {"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}}
    mock_sm.describe_endpoint.side_effect = ClientError(err_response, "DescribeEndpoint")
    mock_sm.delete_endpoint.side_effect = ClientError(err_response, "DeleteEndpoint")

    summary = cleanup_sagemaker_resources(
        endpoint_name="absent-endpoint",
        dry_run=False,
        sagemaker_client=mock_sm,
    )

    assert summary["endpoint_deleted"] is True
    assert len(summary["errors"]) == 0


# ============================================================================
# CloudWatch Telemetry Contracts
# ============================================================================


def test_cloudwatch_telemetry_without_credentials(monkeypatch):
    """Verify log_prediction_cloudwatch falls back to console logging when credentials are absent."""
    # Ensure environment variables are cleared
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    payload = {"age": 55, "bmi": 30.2, "blood_glucose_level": 140}
    result = log_prediction_cloudwatch(payload=payload, risk_score=42.5, latency=12.4)

    assert result is True


def test_cloudwatch_telemetry_with_credentials_mock():
    """Verify CloudWatch telemetry emits metrics and logs when credentials exist."""
    mock_cw = MagicMock()
    mock_logs = MagicMock()

    client = CloudWatchTelemetryClient(
        cw_client=mock_cw,
        logs_client=mock_logs,
    )

    with patch("app.monitoring.cloudwatch.has_aws_credentials", return_value=True):
        payload = {"gender": "Male", "age": 60, "blood_glucose_level": 150}
        result = log_prediction_cloudwatch(
            payload=payload,
            risk_score=78.2,
            latency=15.1,
            client=client,
        )

        assert result is True
        mock_cw.put_metric_data.assert_called_once()
        mock_logs.put_log_events.assert_called_once()

        # Inspect metric call payload
        call_kwargs = mock_cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == "DiabetesMLOps"
        metric_names = [m["MetricName"] for m in call_kwargs["MetricData"]]
        assert "PredictionLatency" in metric_names
        assert "RiskScore" in metric_names


def test_cloudwatch_telemetry_exception_safeguard():
    """Verify log_prediction_cloudwatch never crashes if unexpected exception is raised."""
    bad_client = MagicMock()
    bad_client.emit_metrics.side_effect = RuntimeError("Fatal network crash")

    with patch("app.monitoring.cloudwatch.has_aws_credentials", return_value=True):
        # Should gracefully handle error and fallback without raising
        result = log_prediction_cloudwatch(
            payload={"test": 1},
            risk_score=50.0,
            latency=10.0,
            client=bad_client,
        )
        assert result is True or result is False
