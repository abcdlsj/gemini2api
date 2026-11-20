"""gemini2api - the Gemini web app behind an OpenAI-compatible API."""

from ._version import __version__
from .config import Account, Settings, load
from .server import create_app

__all__ = ["Account", "Settings", "__version__", "create_app", "load"]
