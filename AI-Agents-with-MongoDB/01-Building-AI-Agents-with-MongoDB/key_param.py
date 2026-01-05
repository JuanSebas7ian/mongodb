import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
voyage_api_key = os.getenv("VOYAGE_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")

# AWS Credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")