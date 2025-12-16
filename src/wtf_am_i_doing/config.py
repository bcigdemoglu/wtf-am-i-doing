"""Configuration constants and utility functions."""

import os
import shutil
import subprocess
from pathlib import Path

# Paths
APP_DIR = Path.home() / ".wtf-am-i-doing"
TEMP_DIR = APP_DIR / "temp"
FASTVLM_DIR = APP_DIR / "ml-fastvlm"
CHECKPOINTS_DIR = FASTVLM_DIR / "checkpoints"
MLX_CHECKPOINTS_DIR = FASTVLM_DIR / "checkpoints-mlx"
DEFAULT_DIARY_PATH = APP_DIR / "diary.json"
DEFAULT_ERROR_PATH = APP_DIR / "error.json"
SETTINGS_PATH = APP_DIR / "settings.json"
SETTINGS_DEFAULT_PATH = APP_DIR / "settings.default.json"
DESKTOP_SYMLINK_PATH = Path.home() / "Desktop" / "diary.json"
LOG_FILE = APP_DIR / "app.log"

# Model providers
PROVIDER_FASTVLM = "fastvlm"
PROVIDER_CLAUDE = "claude"

# FastVLM models (display name -> folder name)
FASTVLM_MODELS = {
    "FastVLM-0.5B": "llava-fastvithd_0.5b_stage3",
    "FastVLM-1.5B": "llava-fastvithd_1.5b_stage3",
    "FastVLM-7B": "llava-fastvithd_7b_stage3",
}

# Claude models (display name -> CLI model name)
CLAUDE_MODELS = {
    "Claude Opus": "opus",
    "Claude Sonnet": "sonnet",
    "Claude Haiku": "haiku",
}

# Combined model list for UI (display name -> (provider, identifier))
ALL_MODELS = {
    **{name: (PROVIDER_FASTVLM, folder) for name, folder in FASTVLM_MODELS.items()},
    **{name: (PROVIDER_CLAUDE, model) for name, model in CLAUDE_MODELS.items()},
}

# Legacy alias for backward compatibility
MODELS = FASTVLM_MODELS


def get_model_provider(display_name: str) -> str | None:
    """Get the provider for a model display name."""
    if display_name in ALL_MODELS:
        return ALL_MODELS[display_name][0]
    return None


def get_model_identifier(display_name: str) -> str | None:
    """Get the identifier (folder name or CLI name) for a model."""
    if display_name in ALL_MODELS:
        return ALL_MODELS[display_name][1]
    return None


def get_diary_model_name(display_name: str) -> str:
    """Get a clean model name for diary entries."""
    provider = get_model_provider(display_name)
    if provider == PROVIDER_FASTVLM:
        # fastvlm-0.5b, fastvlm-1.5b, fastvlm-7b
        size = display_name.split("-")[1].lower()
        return f"fastvlm-{size}"
    elif provider == PROVIDER_CLAUDE:
        # claude-opus, claude-sonnet, claude-haiku
        model = display_name.split(" ")[1].lower()
        return f"claude-{model}"
    return display_name.lower()

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
DEFAULT_PROMPT = """Describe the current screen in 6 lines max. Do not guess. If unclear, say "unclear" and keep going.

Format:
Task:
Active: <app or site + page/view>
Focus: <top 2 windows/panels>
Inputs: <what user is typing/selecting, if visible>
Evidence: "<exact quote 1>" | "<exact quote 2>" (only if clearly readable)
Confidence: High/Med/Low (one short reason)"""

# FastVLM repo
FASTVLM_REPO = "apple/ml-fastvlm"


def get_default_settings() -> dict:
    """Get the default settings as a dictionary."""
    return {
        "diary_path": str(DEFAULT_DIARY_PATH),
        "model": "FastVLM-0.5B",
        "interval": DEFAULT_INTERVAL,
        "resolution": "High",
        "prompt": DEFAULT_PROMPT,
        "use_apple_silicon": False,
    }


def ensure_app_dir() -> None:
    """Create the application directory if it doesn't exist."""
    APP_DIR.mkdir(parents=True, exist_ok=True)


def ensure_default_settings() -> None:
    """Create settings.default.json if it doesn't exist."""
    import json
    ensure_app_dir()
    if not SETTINGS_DEFAULT_PATH.exists():
        with open(SETTINGS_DEFAULT_PATH, "w") as f:
            json.dump(get_default_settings(), f, indent=2)


def open_settings_file() -> None:
    """Open the settings file in the default application."""
    if SETTINGS_PATH.exists():
        open_file_in_default_app(SETTINGS_PATH)
    elif SETTINGS_DEFAULT_PATH.exists():
        open_file_in_default_app(SETTINGS_DEFAULT_PATH)


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


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (ARM64)."""
    import platform
    return platform.machine() == "arm64"
