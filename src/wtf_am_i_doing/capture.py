"""Screenshot capture functionality."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import mss
import mss.tools
from PIL import Image

from .config import RESOLUTIONS, TEMP_DIR

logger = logging.getLogger(__name__)


@dataclass
class MonitorInfo:
    """Information about a captured monitor."""
    index: int
    name: str
    width: int
    height: int


@dataclass
class Screenshot:
    """A captured screenshot with metadata."""
    monitor: MonitorInfo
    image_path: Path

    def cleanup(self) -> None:
        """Remove the temporary screenshot file."""
        if self.image_path.exists():
            self.image_path.unlink()
            logger.debug(f"Cleaned up screenshot: {self.image_path.name}")


def cleanup_temp_dir() -> None:
    """Clean up any leftover screenshot files from previous sessions."""
    if not TEMP_DIR.exists():
        return

    count = 0
    for f in TEMP_DIR.glob("wtf_monitor*.png"):
        try:
            f.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {f}: {e}")

    if count > 0:
        logger.info(f"Cleaned up {count} leftover screenshot(s) from temp directory")


def get_monitor_name(index: int) -> str:
    """Get a human-readable name for a monitor."""
    if index == 0:
        return "All Monitors"
    return f"Monitor {index}"


def capture_monitors(resolution: str = "High") -> list[Screenshot]:
    """
    Capture screenshots of all monitors.

    Args:
        resolution: One of "High", "Medium", "Low"

    Returns:
        List of Screenshot objects with paths to temporary image files.
    """
    scale = RESOLUTIONS.get(resolution, 1.0)
    screenshots = []

    with mss.mss() as sct:
        # sct.monitors[0] is "all monitors combined"
        # sct.monitors[1:] are individual monitors
        monitors = sct.monitors[1:]  # Skip the combined view
        logger.info(f"Found {len(monitors)} monitor(s)")

        for idx, monitor in enumerate(monitors, start=1):
            logger.debug(f"Capturing monitor {idx}: {monitor}")
            # Capture the monitor
            sct_img = sct.grab(monitor)
            logger.debug(f"Captured image size: {sct_img.size}")

            # Convert to PIL Image for resizing
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            # Apply resolution scaling
            if scale != 1.0:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to app's temp directory
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            temp_path = TEMP_DIR / f"wtf_monitor{idx}_{int(time.time() * 1000)}.png"
            img.save(str(temp_path), "PNG")

            # Create monitor info
            monitor_info = MonitorInfo(
                index=idx,
                name=get_monitor_name(idx),
                width=monitor["width"],
                height=monitor["height"],
            )

            screenshots.append(Screenshot(
                monitor=monitor_info,
                image_path=temp_path,
            ))

    return screenshots


def get_monitor_count() -> int:
    """Get the number of monitors (excluding the combined view)."""
    with mss.mss() as sct:
        return len(sct.monitors) - 1  # Exclude monitors[0] which is combined


def create_thumbnail(image_path: Path, max_size: tuple[int, int] = (200, 150)) -> Image.Image:
    """
    Create a thumbnail from a screenshot.

    Args:
        image_path: Path to the screenshot file
        max_size: Maximum (width, height) for the thumbnail

    Returns:
        PIL Image object of the thumbnail
    """
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        # Return a copy since the original will be closed
        return img.copy()
