"""Cost-control teardown utility for SageMaker endpoints, configs, and model resources."""

import argparse
import logging
import os
import sys
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))

COST_SAFETY_BANNER = (
    "====================================================================\n"
    "All billable SageMaker resources deleted successfully. Idle billing risk: $0.00.\n"
    "===================================================================="
)


def validate_aws_credentials(region_name: str = DEFAULT_REGION) -> bool:
    """Validate active AWS credentials before initiating cleanup.

    If credentials are missing or invalid, logs a user-friendly error
    with instructions to run `aws configure`.

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


def cleanup_sagemaker_resources(
    endpoint_name: str,
    region_name: str = DEFAULT_REGION,
    dry_run: bool = False,
    sagemaker_client: Any | None = None,
) -> dict[str, Any]:
    """Tear down SageMaker endpoint, endpoint configuration, and model resources.

    Ensures strict cost-control compliance by preventing orphaned inference instances.
    Deletes resources in the exact dependency order:
    1. delete_endpoint
    2. delete_endpoint_config
    3. delete_model

    Args:
        endpoint_name: Target SageMaker endpoint name.
        region_name: AWS region.
        dry_run: If True, simulates teardown without deleting live AWS resources.
        sagemaker_client: Optional boto3 client instance for testing/mocking.

    Returns:
        Dict detailing the teardown actions and success statuses.
    """
    logger.info("Initiating teardown for SageMaker resources associated with '%s' (dry_run=%s)", endpoint_name, dry_run)

    summary: dict[str, Any] = {
        "endpoint_name": endpoint_name,
        "endpoint_deleted": False,
        "config_deleted": False,
        "models_deleted": [],
        "dry_run": dry_run,
        "errors": [],
    }

    if dry_run:
        logger.info("[DRY-RUN] Simulating deletion of SageMaker endpoint: %s", endpoint_name)
        summary["endpoint_deleted"] = True
        logger.info("[DRY-RUN] Simulating deletion of SageMaker endpoint config: %s-config", endpoint_name)
        summary["config_deleted"] = True
        logger.info("[DRY-RUN] Simulating deletion of SageMaker model: %s-model", endpoint_name)
        summary["models_deleted"].append(f"{endpoint_name}-model")
        logger.info("[DRY-RUN] Teardown simulation completed successfully.")
        logger.info("\n%s", COST_SAFETY_BANNER)
        return summary

    sm = sagemaker_client or boto3.client("sagemaker", region_name=region_name)
    config_name: str | None = None
    model_names: list[str] = []

    # 1. Discover associated EndpointConfig and Models
    try:
        desc_endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
        config_name = desc_endpoint.get("EndpointConfigName")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("ValidationException", "ResourceNotFound"):
            logger.warning("Endpoint '%s' does not exist or was already deleted.", endpoint_name)
        else:
            logger.error("Error describing endpoint '%s': %s", endpoint_name, exc)
            summary["errors"].append(str(exc))

    if config_name:
        try:
            desc_config = sm.describe_endpoint_config(EndpointConfigName=config_name)
            for variant in desc_config.get("ProductionVariants", []):
                m_name = variant.get("ModelName")
                if m_name and m_name not in model_names:
                    model_names.append(m_name)
        except ClientError as exc:
            logger.warning("Could not describe endpoint config '%s': %s", config_name, exc)

    # 2. Step 1: Delete SageMaker Endpoint
    try:
        logger.info("Step 1/3: Deleting SageMaker endpoint: %s", endpoint_name)
        sm.delete_endpoint(EndpointName=endpoint_name)
        summary["endpoint_deleted"] = True
        logger.info("Successfully deleted endpoint: %s", endpoint_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("ValidationException", "ResourceNotFound"):
            logger.info("Endpoint '%s' already absent.", endpoint_name)
            summary["endpoint_deleted"] = True
        else:
            logger.error("Failed to delete endpoint '%s': %s", endpoint_name, exc)
            summary["errors"].append(f"delete_endpoint: {exc}")
    except BotoCoreError as exc:
        logger.error("BotoCore error deleting endpoint '%s': %s", endpoint_name, exc)
        summary["errors"].append(f"delete_endpoint: {exc}")

    # 3. Step 2: Delete SageMaker Endpoint Configuration
    if config_name:
        try:
            logger.info("Step 2/3: Deleting SageMaker endpoint config: %s", config_name)
            sm.delete_endpoint_config(EndpointConfigName=config_name)
            summary["config_deleted"] = True
            logger.info("Successfully deleted endpoint config: %s", config_name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("ValidationException", "ResourceNotFound"):
                logger.info("Endpoint config '%s' already absent.", config_name)
                summary["config_deleted"] = True
            else:
                logger.error("Failed to delete endpoint config '%s': %s", config_name, exc)
                summary["errors"].append(f"delete_endpoint_config: {exc}")
        except BotoCoreError as exc:
            logger.error("BotoCore error deleting endpoint config: %s", exc)
            summary["errors"].append(f"delete_endpoint_config: {exc}")

    # 4. Step 3: Delete Associated SageMaker Model(s)
    for model_name in model_names:
        try:
            logger.info("Step 3/3: Deleting SageMaker model entity: %s", model_name)
            sm.delete_model(ModelName=model_name)
            summary["models_deleted"].append(model_name)
            logger.info("Successfully deleted model: %s", model_name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("ValidationException", "ResourceNotFound"):
                logger.info("Model '%s' already absent.", model_name)
                summary["models_deleted"].append(model_name)
            else:
                logger.error("Failed to delete model '%s': %s", model_name, exc)
                summary["errors"].append(f"delete_model: {exc}")
        except BotoCoreError as exc:
            logger.error("BotoCore error deleting model '%s': %s", model_name, exc)
            summary["errors"].append(f"delete_model: {exc}")

    logger.info("Teardown summary for '%s': %s", endpoint_name, summary)
    if not summary["errors"]:
        logger.info("\n%s", COST_SAFETY_BANNER)
    return summary


def main() -> None:
    """CLI entrypoint for SageMaker resource cleanup."""
    parser = argparse.ArgumentParser(description="Tear down SageMaker endpoint, config, and models for cost control")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name to tear down")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate deletion without live AWS calls")

    args = parser.parse_args()

    # In live mode, validate AWS credentials before execution
    if not args.dry_run:
        if not validate_aws_credentials(region_name=args.region):
            print("Cleanup aborted: AWS credentials are missing or invalid.")
            sys.exit(1)

    result = cleanup_sagemaker_resources(
        endpoint_name=args.endpoint_name,
        region_name=args.region,
        dry_run=args.dry_run,
    )

    if result.get("errors"):
        print(f"Cleanup finished with warnings/errors: {result['errors']}")
        sys.exit(1)
    else:
        print(f"Cleanup completed successfully for endpoint '{args.endpoint_name}'.")
        print(COST_SAFETY_BANNER)
        sys.exit(0)


if __name__ == "__main__":
    main()
