"""Settings, read from environment variables (and a local .env file if present)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader so we don't need an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

DB_PATH = Path(os.getenv("FINPILOT_DB", ROOT / "data" / "finpilot.db"))
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"

MODELS_DIR = Path(os.getenv("FINPILOT_MODELS", ROOT / "models"))
KB_DIR = ROOT / "finpilot" / "kb"
REPORTS_DIR = ROOT / "reports"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Below this intent confidence the assistant hands the chat to a human agent.
# The value is chosen from the coverage/accuracy curve in reports/intent_eval.md.
HANDOFF_THRESHOLD = float(os.getenv("FINPILOT_HANDOFF_THRESHOLD", "0.5"))
