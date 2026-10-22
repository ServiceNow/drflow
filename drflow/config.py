"""Load DRFLOW configuration from the environment and the project's .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

# WARNING: We use override=True, which means .env values WILL OVERRIDE
# any existing environment variables from your shell (.zshrc/.bashrc)
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
VLLM_API_URL = os.getenv("VLLM_API_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
VLLM_MODEL = os.getenv("VLLM_MODEL")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")

NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

DRFLOW_DOCKER_IMAGE = os.getenv("DRFLOW_DOCKER_IMAGE", "drbench-services")
DRFLOW_DOCKER_TAG = os.getenv("DRFLOW_DOCKER_TAG", "latest")


def validate_required_keys(required_keys: list[str]) -> bool:
    """
    Validate that required environment variables are set.

    Args:
        required_keys: List of environment variable names that must be set

    Returns:
        True if all required keys are set, False otherwise
    """
    missing_keys = []
    for key in required_keys:
        if not globals().get(key):
            missing_keys.append(key)

    if missing_keys:
        print(f"Warning: Missing required environment variables: {', '.join(missing_keys)}")
        print("Please set these in your .env file or environment")
        return False
    return True
