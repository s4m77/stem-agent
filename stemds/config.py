"""Project configuration defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_SANDBOX_TIMEOUT_SEC = 15.0
