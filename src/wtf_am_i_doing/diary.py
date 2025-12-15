"""JSON diary file management."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capture import MonitorInfo

# Thread lock for file operations (RLock allows same thread to re-acquire)
_file_lock = threading.RLock()


def load_diary(path: Path) -> dict[str, Any]:
    """
    Load an existing diary file or create a new structure.

    Args:
        path: Path to the diary JSON file

    Returns:
        Dictionary with diary structure
    """
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure entries key exists
                if "entries" not in data:
                    data["entries"] = []
                return data
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, start fresh but keep backup
            backup_path = path.with_suffix(".json.backup")
            if path.exists():
                path.rename(backup_path)

    return {"entries": []}


def save_diary(path: Path, data: dict[str, Any]) -> None:
    """
    Save diary data to file atomically.

    Args:
        path: Path to the diary JSON file
        data: Dictionary with diary data
    """
    with _file_lock:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first, then rename (atomic on most systems)
        temp_path = path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.rename(path)


def append_entry(
    path: Path,
    model: str,
    description: str,
    monitor: MonitorInfo | None = None,
    is_single_monitor: bool = True,
    inference_time_ms: int | None = None,
) -> dict[str, Any]:
    """
    Append a new entry to the diary.

    Args:
        path: Path to the diary JSON file
        model: Name of the model used
        description: VLM description of the screenshot
        monitor: Monitor information (None if single monitor)
        is_single_monitor: Whether this is a single-monitor setup
        inference_time_ms: Time taken for inference in milliseconds

    Returns:
        The entry that was added
    """
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "description": description,
    }

    if inference_time_ms is not None:
        entry["inference_ms"] = inference_time_ms

    # Only include monitor info for multi-monitor setups
    if not is_single_monitor and monitor is not None:
        entry["monitor"] = monitor.index
        entry["monitor_name"] = monitor.name

    with _file_lock:
        data = load_diary(path)
        data["entries"].append(entry)
        save_diary(path, data)

    return entry


def append_error(
    error_path: Path,
    model: str,
    error_message: str,
    monitor: MonitorInfo | None = None,
    is_single_monitor: bool = True,
) -> dict[str, Any]:
    """
    Append an error entry to the error log file.

    Args:
        error_path: Path to the error JSON file
        model: Name of the model used
        error_message: Description of the error
        monitor: Monitor information (None if single monitor)
        is_single_monitor: Whether this is a single-monitor setup

    Returns:
        The error entry that was added
    """
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "error": error_message,
    }

    # Only include monitor info for multi-monitor setups
    if not is_single_monitor and monitor is not None:
        entry["monitor"] = monitor.index
        entry["monitor_name"] = monitor.name

    with _file_lock:
        data = load_diary(error_path)
        data["entries"].append(entry)
        save_diary(error_path, data)

    return entry


def get_entry_count(path: Path) -> int:
    """Get the total number of entries in the diary."""
    if not path.exists():
        return 0

    data = load_diary(path)
    return len(data.get("entries", []))


def get_error_count(error_path: Path) -> int:
    """Get the number of error entries in the error file."""
    if not error_path.exists():
        return 0

    data = load_diary(error_path)
    return len(data.get("entries", []))


def get_last_entry(path: Path) -> dict[str, Any] | None:
    """Get the most recent entry from the diary."""
    if not path.exists():
        return None

    data = load_diary(path)
    entries = data.get("entries", [])

    if entries:
        return entries[-1]
    return None
