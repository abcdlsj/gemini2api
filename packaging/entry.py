"""Entry point for the frozen binary: `gemini2api` with the same argv."""

import sys

from gemini2api.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
