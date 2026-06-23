import os
from pathlib import Path

_APP_DIR = Path(__file__).parent   # norm-ap-labeling-ui/app/
_APP_ROOT = _APP_DIR.parent        # norm-ap-labeling-ui/

# Load .env from the app root if present (override default data paths without editing this file)
_dotenv_path = _APP_ROOT / ".env"
if _dotenv_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path, override=False)
    except ImportError:
        for _line in _dotenv_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

_TELECOM_DIR = _APP_ROOT.parent / "data" / "telecom"  # fallback default

RESOURCES_DIR = _APP_ROOT / "resources"

# Data source paths — override with env vars
DEFAULT_CONVERSATIONS_PATH = os.environ.get(
    "CONVERSATIONS_PATH",
    str(_TELECOM_DIR / "conversations.json"),
)
DEFAULT_PREDICTIONS_PATH = os.environ.get(
    "PREDICTIONS_PATH",
    str(_TELECOM_DIR / "predictions" / "ap_predictions.json"),
)
DEFAULT_NORMS_PATH = os.environ.get(
    "NORMS_PATH",
    str(_TELECOM_DIR / "norms.json"),
)
DEFAULT_PROPS_PATH = os.environ.get(
    "PROPS_PATH",
    str(_TELECOM_DIR / "propositions.json"),
)

# Deepseek labeler ID for telecom ap_predictions.json (deepseek-r1-0528/sensor/-/none)
DEEPSEEK_LABELER_ID = "a618d762-1ecf-5d62-9f14-f29c0c2381ae"

# norm_compliance package lives inside the app root
NORM_COMPLIANCE_REPO = str(_APP_ROOT)

# Storage
LABELS_DIR = RESOURCES_DIR / "labels"
JOBS_DIR = RESOURCES_DIR / "jobs"
USERS_FILE = RESOURCES_DIR / "users.jsonl"
