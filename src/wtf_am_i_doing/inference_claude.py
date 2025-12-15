"""Claude CLI inference backend."""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .config import CLAUDE_MODELS

logger = logging.getLogger(__name__)


def _get_env_with_path() -> dict:
    """Get environment with common paths added for node/claude."""
    env = os.environ.copy()

    # Add common paths where node might be installed
    extra_paths = [
        "/usr/local/bin",
        "/opt/homebrew/bin",
        str(Path.home() / ".nvm" / "versions" / "node" / "v20.0.0" / "bin"),  # nvm
        str(Path.home() / ".volta" / "bin"),  # volta
        str(Path.home() / ".local" / "bin"),
        "/usr/bin",
    ]

    current_path = env.get("PATH", "")
    env["PATH"] = ":".join(extra_paths) + ":" + current_path

    return env


class ClaudeSetupError(Exception):
    """Raised when Claude CLI setup fails."""
    pass


class ClaudeInferenceError(Exception):
    """Raised when Claude inference fails."""
    pass


def _find_claude() -> str | None:
    """Find the claude CLI executable."""
    logger.debug("Looking for claude CLI...")

    # First try PATH
    claude_path = shutil.which("claude")
    if claude_path:
        logger.debug(f"Found claude in PATH: {claude_path}")
        return claude_path

    # Check common locations
    common_paths = [
        Path.home() / ".claude" / "local" / "claude",  # Claude Code default
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
        Path.home() / ".local" / "bin" / "claude",
        Path.home() / "bin" / "claude",
    ]

    for p in common_paths:
        p = Path(p)
        logger.debug(f"Checking: {p}")
        if p.exists():
            logger.info(f"Found claude at: {p}")
            return str(p)

    logger.error("Could not find claude CLI")
    return None


def is_claude_installed() -> bool:
    """Check if claude CLI is installed."""
    return _find_claude() is not None


def is_model_available(model_name: str) -> bool:
    """
    Check if a Claude model is available.
    Claude models are always available if the CLI is installed.
    """
    return model_name in CLAUDE_MODELS and is_claude_installed()


def run_inference(
    image_path: Path,
    model_name: str,
    prompt: str,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """
    Run Claude inference on an image using the claude CLI.

    Args:
        image_path: Path to the image file
        model_name: Display name of the model (e.g., "Claude Opus")
        prompt: Text prompt for the model
        progress_callback: Optional callback for progress updates

    Returns:
        The model's text output

    Raises:
        ClaudeInferenceError: If inference fails
    """
    claude_model = CLAUDE_MODELS.get(model_name)
    if not claude_model:
        raise ClaudeInferenceError(f"Unknown Claude model: {model_name}")

    claude = _find_claude()
    if not claude:
        raise ClaudeInferenceError(
            "Claude CLI not found. Please install it first.\n\n"
            "Install with: npm install -g @anthropic-ai/claude-code"
        )

    if progress_callback:
        progress_callback(f"Running inference with {model_name}...")

    try:
        # Run from the screenshot directory, use just the filename
        image_dir = image_path.parent
        image_name = image_path.name
        full_prompt = f"{prompt} {image_name}"

        # Run: echo "<prompt> <file>" | claude --model=<model> -p
        cmd = [claude, "--model", claude_model, "-p"]
        logger.info(f"Running Claude inference: {' '.join(cmd)}")
        logger.debug(f"Prompt: {full_prompt}")
        logger.debug(f"Working directory: {image_dir}")

        # Get environment with proper PATH for node
        env = _get_env_with_path()
        logger.debug(f"PATH: {env.get('PATH', '')[:200]}...")

        result = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for Claude
            env=env,
            cwd=str(image_dir),  # Run from screenshot directory
        )

        logger.debug(f"Claude stdout: {result.stdout}")
        logger.debug(f"Claude stderr: {result.stderr}")
        logger.debug(f"Claude returncode: {result.returncode}")

        if result.returncode != 0:
            logger.error(f"Claude inference failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise ClaudeInferenceError(f"Claude inference failed: {result.stderr}")

        output = result.stdout.strip()

        if not output:
            logger.error("Claude returned empty output")
            raise ClaudeInferenceError("Claude returned empty output.")

        logger.info(f"Claude inference successful, output length: {len(output)}")
        return output

    except subprocess.TimeoutExpired:
        logger.error("Claude inference timed out after 2 minutes")
        raise ClaudeInferenceError("Claude inference timed out after 2 minutes.")
    except ClaudeInferenceError:
        raise
    except Exception as e:
        logger.exception(f"Claude inference error: {str(e)}")
        raise ClaudeInferenceError(f"Claude inference error: {str(e)}")
