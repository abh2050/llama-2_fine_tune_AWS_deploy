import boto3
import os
import logging
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

def download_model_from_s3(bucket_name, s3_prefix, local_path):
    """Download model files from S3 to local path"""
    try:
        s3 = boto3.client('s3')
        
        # Create local directory
        os.makedirs(local_path, exist_ok=True)
        logger.info(f"Created local directory: {local_path}")
        
        # List objects in S3 bucket with prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
        
        files_downloaded = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Extract filename from S3 key
                    relative_path = obj['Key'].replace(s3_prefix, '')
                    if relative_path and not relative_path.endswith('/'):  # Skip directories
                        local_file_path = os.path.join(local_path, relative_path)
                        
                        # Create subdirectories if needed
                        local_dir = os.path.dirname(local_file_path)
                        if local_dir:
                            os.makedirs(local_dir, exist_ok=True)
                        
                        logger.info(f"Downloading {obj['Key']} to {local_file_path}")
                        s3.download_file(bucket_name, obj['Key'], local_file_path)
                        files_downloaded += 1
        
        if files_downloaded == 0:
            raise ValueError(f"No files found in s3://{bucket_name}/{s3_prefix}")
        
        logger.info(f"Model downloaded successfully to {local_path} ({files_downloaded} files)")
        
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        raise
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            logger.error(f"S3 bucket '{bucket_name}' does not exist")
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied to S3 bucket '{bucket_name}'")
        else:
            logger.error(f"AWS error downloading from S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading from S3: {e}")
        raise

def upload_model_to_s3(local_path, bucket_name, s3_prefix):
    """Upload model files to S3"""
    try:
        s3 = boto3.client('s3')
        
        # Check if bucket exists
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{bucket_name}' does not exist")
                raise
        
        files_uploaded = 0
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')  # Ensure forward slashes
                
                logger.info(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
                s3.upload_file(local_file_path, bucket_name, s3_key)
                files_uploaded += 1
        
        logger.info(f"Model uploaded successfully to s3://{bucket_name}/{s3_prefix} ({files_uploaded} files)")
        
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        raise
    except ClientError as e:
        logger.error(f"AWS error uploading to S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading to S3: {e}")
        raise

def check_s3_model_exists(bucket_name, s3_prefix):
    """Check if model exists in S3"""
    try:
        s3 = boto3.client('s3')
        
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix, MaxKeys=1)
        return 'Contents' in response and len(response['Contents']) > 0
        
    except ClientError as e:
        logger.error(f"Error checking S3 model existence: {e}")
        return False

def create_s3_bucket_if_not_exists(bucket_name, region='us-east-1'):
    """Create S3 bucket if it doesn't exist"""
    try:
        s3 = boto3.client('s3', region_name=region)
        
        # Check if bucket exists
        try:
            s3.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket '{bucket_name}' already exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code != '404':
                raise
        
        # Create bucket
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        logger.info(f"S3 bucket '{bucket_name}' created successfully")
        return True
        
    except ClientError as e:
        logger.error(f"Error creating S3 bucket: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating S3 bucket: {e}")
        raise
