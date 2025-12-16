"""Inference dispatcher - routes to appropriate backend based on model."""

import logging
from pathlib import Path
from typing import Callable

from .config import (
    PROVIDER_FASTVLM,
    PROVIDER_CLAUDE,
    get_model_provider,
    is_model_downloaded,
)
from . import inference_fastvlm
from . import inference_claude
from . import inference_mlx

logger = logging.getLogger(__name__)


# Re-export exceptions for backward compatibility
class SetupError(Exception):
    """Raised when setup/download fails."""
    pass


class InferenceError(Exception):
    """Raised when inference fails."""
    pass


def is_model_available(model_name: str) -> bool:
    """
    Check if a model is available for use.

    Args:
        model_name: Display name of the model

    Returns:
        True if the model is ready to use
    """
    provider = get_model_provider(model_name)

    if provider == PROVIDER_FASTVLM:
        return is_model_downloaded(model_name)
    elif provider == PROVIDER_CLAUDE:
        return inference_claude.is_claude_installed()

    return False


def run_inference(
    image_path: Path,
    model_name: str,
    prompt: str,
    progress_callback: Callable[[str], None] | None = None,
    use_mlx: bool = False,
) -> str:
    """
    Run inference on an image using the specified model.

    Args:
        image_path: Path to the image file
        model_name: Display name of the model (e.g., "FastVLM-0.5B" or "Claude Opus")
        prompt: Text prompt for the model
        progress_callback: Optional callback for progress updates
        use_mlx: If True, use MLX backend for FastVLM models (Apple Silicon only)

    Returns:
        The model's text output

    Raises:
        InferenceError: If inference fails
    """
    provider = get_model_provider(model_name)
    logger.info(f"Running inference with model '{model_name}' (provider: {provider}, mlx: {use_mlx})")

    try:
        if provider == PROVIDER_FASTVLM:
            if use_mlx:
                return inference_mlx.run_inference(
                    image_path, model_name, prompt, progress_callback
                )
            else:
                return inference_fastvlm.run_inference(
                    image_path, model_name, prompt, progress_callback
                )
        elif provider == PROVIDER_CLAUDE:
            return inference_claude.run_inference(
                image_path, model_name, prompt, progress_callback
            )
        else:
            raise InferenceError(f"Unknown model provider for '{model_name}'")

    except inference_fastvlm.FastVLMInferenceError as e:
        raise InferenceError(str(e))
    except inference_claude.ClaudeInferenceError as e:
        raise InferenceError(str(e))
    except inference_mlx.MLXInferenceError as e:
        raise InferenceError(str(e))


# FastVLM setup functions (re-exported for backward compatibility)
def clone_fastvlm(progress_callback: Callable[[str], None] | None = None) -> None:
    """Clone the FastVLM repository."""
    try:
        inference_fastvlm.clone_fastvlm(progress_callback)
    except inference_fastvlm.FastVLMSetupError as e:
        raise SetupError(str(e))


def install_fastvlm_deps(progress_callback: Callable[[str], None] | None = None) -> None:
    """Install FastVLM dependencies."""
    try:
        inference_fastvlm.install_fastvlm_deps(progress_callback)
    except inference_fastvlm.FastVLMSetupError as e:
        raise SetupError(str(e))


def download_models(progress_callback: Callable[[str], None] | None = None) -> None:
    """Download FastVLM model checkpoints."""
    try:
        inference_fastvlm.download_models(progress_callback)
    except inference_fastvlm.FastVLMSetupError as e:
        raise SetupError(str(e))


def full_setup(progress_callback: Callable[[str], None] | None = None) -> None:
    """Perform full FastVLM setup."""
    try:
        inference_fastvlm.full_setup(progress_callback)
    except inference_fastvlm.FastVLMSetupError as e:
        raise SetupError(str(e))


# MLX setup functions
def is_mlx_available() -> bool:
    """Check if MLX is available (Apple Silicon only)."""
    return inference_mlx.is_mlx_available()


def is_mlx_setup_complete() -> bool:
    """Check if MLX setup is complete (mlx-vlm installed + models converted)."""
    return inference_mlx.is_mlx_setup_complete()


def setup_mlx(progress_callback: Callable[[str], None] | None = None) -> None:
    """Setup MLX for Apple Silicon inference."""
    try:
        inference_mlx.setup_mlx(progress_callback)
    except inference_mlx.MLXSetupError as e:
        raise SetupError(str(e))
