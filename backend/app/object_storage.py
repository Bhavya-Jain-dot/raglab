"""S3/MinIO object storage adapter for original ingested files."""
from __future__ import annotations

import os
import uuid


def _client():
    endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("MINIO_ENDPOINT")
    if not endpoint:
        return None
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("S3-compatible storage is configured but boto3 is not installed") from exc
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER", "raglab"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD", ""),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"}),
    )


def store_original(project_id: int, filename: str, body: bytes, content_type: str | None = None) -> str | None:
    client = _client()
    if client is None:
        return None
    bucket = os.getenv("S3_BUCKET") or os.getenv("MINIO_BUCKET", "raglab-artifacts")
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        try:
            client.create_bucket(Bucket=bucket)
        except Exception:
            client.head_bucket(Bucket=bucket)
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    key = f"projects/{project_id}/documents/{uuid.uuid4().hex}.{suffix[:12]}"
    options = {"Bucket": bucket, "Key": key, "Body": body}
    if content_type:
        options["ContentType"] = content_type[:128]
    client.put_object(**options)
    return f"s3://{bucket}/{key}"


def delete_object(storage_path: str | None) -> None:
    if not storage_path or not storage_path.startswith("s3://"):
        return
    client = _client()
    if client is None:
        return
    bucket, _, key = storage_path[5:].partition("/")
    if bucket and key:
        client.delete_object(Bucket=bucket, Key=key)
