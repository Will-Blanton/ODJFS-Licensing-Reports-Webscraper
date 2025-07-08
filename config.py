from dotenv import load_dotenv
import os

load_dotenv()

def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value

# DOC_PROCESSOR_KEY = require_env("DOC_PROCESSOR_KEY")
DOC_PROCESSOR_ID = require_env("DOC_PROCESSOR_ID")
PROJECT_ID = require_env("PROJECT_ID")
GCS_BUCKET = require_env("GCS_BUCKET")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = require_env("CRED_FILE")
