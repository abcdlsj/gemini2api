"""gemini2api - the Gemini web app behind an OpenAI-compatible API."""

from .config import Account, Settings, load
from .server import create_app

__all__ = ["Account", "Settings", "create_app", "load"]
__version__ = "0.1.0"
