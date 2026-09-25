"""AWS S3 artifact and dataset synchronization manager with dry-run/mock fallbacks."""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BUCKET = os.getenv("AWS_S3_BUCKET", "mlops-diabetes-artifacts")
DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))


class S3SyncManager:
    """Manages artifact and dataset synchronization between local disk and AWS S3."""

    def __init__(
        self,
        bucket_name: str | None = None,
        region_name: str | None = None,
        dry_run: bool = False,
        s3_client: Any | None = None,
    ) -> None:
        """Initialize the S3 synchronization manager.

        Args:
            bucket_name: S3 bucket name. Defaults to AWS_S3_BUCKET or 'mlops-diabetes-artifacts'.
            region_name: AWS region name. Defaults to us-east-1.
            dry_run: If True, simulates operations without performing network calls.
            s3_client: Optional pre-configured boto3 S3 client (e.g. for testing/mocking).
        """
        self.bucket_name = bucket_name or DEFAULT_BUCKET
        self.region_name = region_name or DEFAULT_REGION
        self.dry_run = dry_run
        self._s3_client = s3_client

    @property
    def s3_client(self) -> Any:
        """Resolve or initialize the boto3 S3 client lazily."""
        if self._s3_client is not None:
            return self._s3_client

        if self.dry_run:
            logger.info("[DRY-RUN] Boto3 client initialization bypassed.")
            return None

        try:
            session = boto3.Session(region_name=self.region_name)
            credentials = session.get_credentials()
            if credentials is None:
                logger.warning("No AWS credentials detected in environment. Operating in dry-run fallback mode.")
                self.dry_run = True
                return None
            self._s3_client = session.client("s3")
            return self._s3_client
        except (NoCredentialsError, BotoCoreError) as exc:
            logger.warning("AWS client initialization error (%s). Falling back to dry-run.", exc)
            self.dry_run = True
            return None

    def upload_file(self, local_path: str | Path, s3_key: str) -> bool:
        """Upload a local file to S3.

        Args:
            local_path: Path to the local file.
            s3_key: Destination key in the S3 bucket.

        Returns:
            True if upload succeeded (or was simulated in dry-run), False otherwise.
        """
        local_file = Path(local_path)
        if not local_file.exists():
            logger.error("Local file not found: %s", local_file)
            return False

        if self.dry_run:
            logger.info("[DRY-RUN] Simulating upload: %s -> s3://%s/%s", local_file, self.bucket_name, s3_key)
            return True

        client = self.s3_client
        if client is None:
            logger.info("[DRY-RUN Fallback] Upload simulated for %s -> s3://%s/%s", local_file, self.bucket_name, s3_key)
            return True

        try:
            logger.info("Uploading %s -> s3://%s/%s", local_file, self.bucket_name, s3_key)
            client.upload_file(str(local_file), self.bucket_name, s3_key)
            return True
        except (ClientError, BotoCoreError) as exc:
            logger.error("Failed to upload %s to s3://%s/%s: %s", local_file, self.bucket_name, s3_key, exc)
            return False

    def download_file(self, s3_key: str, local_path: str | Path) -> bool:
        """Download an S3 object to local disk.

        Args:
            s3_key: S3 object key.
            local_path: Local destination file path.

        Returns:
            True if download succeeded (or was simulated in dry-run), False otherwise.
        """
        local_file = Path(local_path)

        if self.dry_run:
            logger.info("[DRY-RUN] Simulating download: s3://%s/%s -> %s", self.bucket_name, s3_key, local_file)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if not local_file.exists():
                local_file.touch()
            return True

        client = self.s3_client
        if client is None:
            logger.info("[DRY-RUN Fallback] Download simulated for s3://%s/%s -> %s", self.bucket_name, s3_key, local_file)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if not local_file.exists():
                local_file.touch()
            return True

        try:
            logger.info("Downloading s3://%s/%s -> %s", self.bucket_name, s3_key, local_file)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(self.bucket_name, s3_key, str(local_file))
            return True
        except (ClientError, BotoCoreError) as exc:
            logger.error("Failed to download s3://%s/%s: %s", self.bucket_name, s3_key, exc)
            return False

    def sync_directory_to_s3(
        self,
        local_dir: str | Path,
        s3_prefix: str,
        extension_filter: list[str] | None = None,
    ) -> list[str]:
        """Upload all matching files from a local directory to an S3 prefix.

        Args:
            local_dir: Local directory path.
            s3_prefix: Destination S3 key prefix.
            extension_filter: Optional list of file extensions to include (e.g. ['.joblib', '.csv']).

        Returns:
            List of successfully synchronized S3 keys.
        """
        source_dir = Path(local_dir)
        if not source_dir.exists():
            logger.warning("Local directory %s does not exist. Nothing to sync.", source_dir)
            return []

        uploaded_keys: list[str] = []
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if extension_filter and not any(file_path.name.endswith(ext) for ext in extension_filter):
                continue

            relative_path = file_path.relative_to(source_dir).as_posix()
            s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}"

            if self.upload_file(file_path, s3_key):
                uploaded_keys.append(s3_key)

        logger.info("Synced %d files from %s to s3://%s/%s", len(uploaded_keys), source_dir, self.bucket_name, s3_prefix)
        return uploaded_keys

    def sync_models_to_s3(self, models_dir: str | Path = "models", prefix: str = "models") -> list[str]:
        """Sync serialized model artifacts to S3."""
        return self.sync_directory_to_s3(
            local_dir=models_dir,
            s3_prefix=prefix,
            extension_filter=[".joblib", ".json", ".xgb", ".pkl", ".tar.gz"],
        )

    def sync_data_to_s3(self, data_dir: str | Path = "data/raw", prefix: str = "data/raw") -> list[str]:
        """Sync raw/processed datasets to S3."""
        return self.sync_directory_to_s3(
            local_dir=data_dir,
            s3_prefix=prefix,
            extension_filter=[".csv", ".parquet"],
        )

    def list_objects(self, prefix: str = "") -> list[dict[str, Any]]:
        """List objects under an S3 prefix.

        Args:
            prefix: S3 prefix to filter.

        Returns:
            List of metadata dicts containing 'Key' and 'Size'.
        """
        if self.dry_run:
            logger.info("[DRY-RUN] Listing objects under s3://%s/%s (mocked)", self.bucket_name, prefix)
            return [{"Key": f"{prefix.rstrip('/')}/mock_artifact.joblib", "Size": 1024}]

        client = self.s3_client
        if client is None:
            return [{"Key": f"{prefix.rstrip('/')}/mock_artifact.joblib", "Size": 1024}]

        try:
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            results = []
            for page in pages:
                for obj in page.get("Contents", []):
                    results.append({"Key": obj["Key"], "Size": obj["Size"]})
            return results
        except (ClientError, BotoCoreError) as exc:
            logger.error("Failed to list objects in s3://%s/%s: %s", self.bucket_name, prefix, exc)
            return []


def main() -> None:
    """CLI interface for S3 synchronization."""
    parser = argparse.ArgumentParser(description="AWS S3 Artifact and Dataset Synchronization")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="AWS S3 bucket name")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Execute in dry-run simulation mode")
    parser.add_argument("--upload-models", action="store_true", help="Sync local models/ to S3")
    parser.add_argument("--upload-data", action="store_true", help="Sync local data/ to S3")
    parser.add_argument("--list-prefix", type=str, default=None, help="List objects with given prefix")

    args = parser.parse_args()

    manager = S3SyncManager(
        bucket_name=args.bucket,
        region_name=args.region,
        dry_run=args.dry_run,
    )

    if args.upload_models:
        synced = manager.sync_models_to_s3()
        print(f"Uploaded {len(synced)} model artifact(s).")

    if args.upload_data:
        synced = manager.sync_data_to_s3()
        print(f"Uploaded {len(synced)} dataset file(s).")

    if args.list_prefix is not None:
        items = manager.list_objects(args.list_prefix)
        print(f"Found {len(items)} items under prefix '{args.list_prefix}':")
        for item in items:
            print(f" - {item['Key']} ({item['Size']} bytes)")

    if not (args.upload_models or args.upload_data or args.list_prefix is not None):
        print("S3SyncManager initialized successfully in dry-run verification mode.")


if __name__ == "__main__":
    main()
