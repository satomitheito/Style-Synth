import boto3
from botocore.client import Config

async def upload_file_to_s3(file, bucket: str, key: str) -> str:
    s3_client = boto3.client("s3", config=Config(signature_version="s3v4"))

    body = await file.read()

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=file.content_type,
    )

    return f"s3://{bucket}/{key}"
