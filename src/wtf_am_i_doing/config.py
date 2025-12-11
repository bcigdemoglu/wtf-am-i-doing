"""Configuration constants and utility functions."""

import os
import shutil
import subprocess
from pathlib import Path

# Paths
APP_DIR = Path.home() / ".wtf-am-i-doing"
FASTVLM_DIR = APP_DIR / "ml-fastvlm"
CHECKPOINTS_DIR = FASTVLM_DIR / "checkpoints"
DEFAULT_DIARY_PATH = APP_DIR / "diary.json"
DESKTOP_SYMLINK_PATH = Path.home() / "Desktop" / "diary.json"
LOG_FILE = APP_DIR / "app.log"

# FastVLM models
MODELS = {
    "FastVLM-0.5B": "fastvlm_0.5b_stage3",
    "FastVLM-1.5B": "fastvlm_1.5b_stage3",
    "FastVLM-7B": "fastvlm_7b_stage3",
}

# Interval settings (in seconds)
MIN_INTERVAL = 5
MAX_INTERVAL = 300
DEFAULT_INTERVAL = 10

# Resolution settings (scale factors)
RESOLUTIONS = {
    "High": 1.0,
    "Medium": 0.5,
    "Low": 0.25,
}

# Default prompt
DEFAULT_PROMPT = "Describe what the user is doing on their computer screen."

# FastVLM repo
FASTVLM_REPO = "apple/ml-fastvlm"


def ensure_app_dir() -> None:
    """Create the application directory if it doesn't exist."""
    APP_DIR.mkdir(parents=True, exist_ok=True)


def is_gh_installed() -> bool:
    """Check if GitHub CLI is installed."""
    return shutil.which("gh") is not None


def is_uv_installed() -> bool:
    """Check if uv is installed."""
    return shutil.which("uv") is not None


def is_fastvlm_cloned() -> bool:
    """Check if FastVLM repo has been cloned."""
    return (FASTVLM_DIR / "predict.py").exists()


def is_model_downloaded(model_name: str) -> bool:
    """Check if a specific model has been downloaded."""
    model_dir = CHECKPOINTS_DIR / MODELS.get(model_name, "")
    return model_dir.exists() and any(model_dir.iterdir())


def get_available_models() -> list[str]:
    """Get list of downloaded models."""
    available = []
    for display_name, folder_name in MODELS.items():
        if is_model_downloaded(display_name):
            available.append(display_name)
    return available


def check_screen_recording_permission() -> bool:
    """
    Check if screen recording permission is granted.
    Returns True if we can capture, False otherwise.
    """
    try:
        import mss
        with mss.mss() as sct:
            # Try to capture a small region
            sct.grab(sct.monitors[0])
        return True
    except Exception:
        return False


def open_screen_recording_settings() -> None:
    """Open System Preferences to Screen Recording settings."""
    subprocess.run([
        "open",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
    ])


def open_file_in_default_app(file_path: Path) -> None:
    """Open a file with the system default application."""
    subprocess.run(["open", str(file_path)])
