"""Load optional project secrets from .env at the repo root."""
from pathlib import Path


def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
