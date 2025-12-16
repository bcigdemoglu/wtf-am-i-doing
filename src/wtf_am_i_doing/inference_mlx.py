"""MLX inference backend for Apple Silicon."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .config import (
    CHECKPOINTS_DIR,
    FASTVLM_DIR,
    FASTVLM_MODELS,
    MLX_CHECKPOINTS_DIR,
    is_apple_silicon,
)

logger = logging.getLogger(__name__)

CONDA_ENV_NAME = "fastvlm"


class MLXSetupError(Exception):
    """Raised when MLX setup fails."""
    pass


class MLXInferenceError(Exception):
    """Raised when MLX inference fails."""
    pass


def _find_conda() -> str | None:
    """Find the conda executable, checking common locations."""
    # First try PATH
    conda_path = shutil.which("conda")
    if conda_path:
        return conda_path

    # Check common conda locations
    common_paths = [
        "/usr/local/bin/conda",
        "/opt/homebrew/bin/conda",
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path.home() / "miniforge3" / "bin" / "conda",
        "/usr/local/Caskroom/miniconda/base/bin/conda",
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda",
    ]

    for p in common_paths:
        p = Path(p)
        if p.exists():
            return str(p)

    return None


def _conda_env_exists() -> bool:
    """Check if the fastvlm conda environment exists."""
    conda = _find_conda()
    if not conda:
        return False
    try:
        result = subprocess.run(
            [conda, "env", "list"],
            capture_output=True,
            text=True,
        )
        return CONDA_ENV_NAME in result.stdout
    except Exception:
        return False


def is_mlx_available() -> bool:
    """Check if MLX inference is available (Apple Silicon only)."""
    return is_apple_silicon()


def is_mlx_installed() -> bool:
    """Check if mlx-vlm is installed in the conda environment."""
    if not _conda_env_exists():
        return False

    conda = _find_conda()
    if not conda:
        return False

    try:
        result = subprocess.run(
            [conda, "run", "-n", CONDA_ENV_NAME, "python", "-c", "import mlx_vlm"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_model_converted(model_name: str) -> bool:
    """Check if a specific model has been converted to MLX format."""
    model_folder = FASTVLM_MODELS.get(model_name)
    if not model_folder:
        return False

    mlx_model_dir = MLX_CHECKPOINTS_DIR / model_folder
    # Check for config.json which indicates a converted model
    return (mlx_model_dir / "config.json").exists()


def are_all_models_converted() -> bool:
    """Check if all FastVLM models have been converted to MLX format."""
    for model_name in FASTVLM_MODELS:
        if not is_model_converted(model_name):
            return False
    return True


def is_mlx_setup_complete() -> bool:
    """Check if MLX setup is complete (mlx-vlm installed + models converted)."""
    return is_mlx_installed() and are_all_models_converted()


def setup_mlx(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Setup MLX by opening a Terminal window to install mlx-vlm and convert models.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        MLXSetupError: If setup fails
    """
    if not _conda_env_exists():
        raise MLXSetupError(
            "FastVLM conda environment not found.\n\n"
            "Please click 'Download FastVLM' first to set up the environment."
        )

    # Check if PyTorch models exist (needed for conversion)
    pytorch_models_exist = False
    for model_folder in FASTVLM_MODELS.values():
        if (CHECKPOINTS_DIR / model_folder).exists():
            pytorch_models_exist = True
            break

    if not pytorch_models_exist:
        raise MLXSetupError(
            "No FastVLM models found.\n\n"
            "Please click 'Download FastVLM' first to download the models."
        )

    if progress_callback:
        progress_callback("Opening Terminal to setup Apple Silicon...")

    try:
        # Create a wrapper script for MLX setup
        wrapper_script = FASTVLM_DIR / "setup_mlx_wrapper.sh"

        conda = _find_conda()
        if not conda:
            raise MLXSetupError("Could not find conda.")

        # Build the conversion commands for each model
        convert_commands = []
        for model_name, model_folder in FASTVLM_MODELS.items():
            pytorch_path = CHECKPOINTS_DIR / model_folder
            mlx_path = MLX_CHECKPOINTS_DIR / model_folder
            if pytorch_path.exists():
                convert_commands.append(f'''
echo ""
echo "Converting {model_name}..."
"{conda}" run -n {CONDA_ENV_NAME} python -m mlx_vlm.convert \\
    --hf-path "{pytorch_path}" \\
    --mlx-path "{mlx_path}"
''')

        convert_script = "\n".join(convert_commands)

        wrapper_script.write_text(f'''#!/bin/bash
echo "========================================="
echo "  Setting up Apple Silicon (MLX)"
echo "========================================="
echo ""

# Check if already set up
if "{conda}" run -n {CONDA_ENV_NAME} python -c "import mlx_vlm" 2>/dev/null; then
    echo "mlx-vlm is already installed."
else
    echo "Installing mlx-vlm..."
    "{conda}" run -n {CONDA_ENV_NAME} pip install mlx-vlm
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to install mlx-vlm"
        echo "Press any key to exit..."
        read -n 1
        exit 1
    fi
    echo "mlx-vlm installed successfully."
fi

# Check if models already converted
MLX_DIR="{MLX_CHECKPOINTS_DIR}"
if [ -d "$MLX_DIR" ] && [ "$(ls -A "$MLX_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "WARNING: MLX models already exist!"
    ls -la "$MLX_DIR"
    echo ""
    read -p "Do you want to re-convert? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Setup complete (using existing models)."
        echo "Press any key to exit..."
        read -n 1
        exit 0
    fi
fi

echo ""
echo "Converting models to MLX format..."
echo "This may take a few minutes per model."

mkdir -p "{MLX_CHECKPOINTS_DIR}"
{convert_script}

echo ""
echo "========================================="
echo "  Apple Silicon setup complete!"
echo "========================================="
echo ""
echo "You can now enable 'Use Apple Silicon' in the app."
echo "Press any key to exit..."
read -n 1
''')
        wrapper_script.chmod(0o755)

        # Open Terminal.app with the wrapper script
        subprocess.run([
            "open", "-a", "Terminal", str(wrapper_script)
        ], check=True)

        if progress_callback:
            progress_callback("Check Terminal for setup progress")

    except subprocess.CalledProcessError as e:
        raise MLXSetupError(f"Failed to open Terminal: {e}")
    except Exception as e:
        raise MLXSetupError(f"Failed to setup MLX: {str(e)}")


def run_inference(
    image_path: Path,
    model_name: str,
    prompt: str,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """
    Run MLX inference on an image using mlx_vlm.generate.

    Args:
        image_path: Path to the image file
        model_name: Display name of the model (e.g., "FastVLM-0.5B")
        prompt: Text prompt for the model
        progress_callback: Optional callback for progress updates

    Returns:
        The model's text output

    Raises:
        MLXInferenceError: If inference fails
    """
    if not is_apple_silicon():
        raise MLXInferenceError("MLX is only available on Apple Silicon Macs.")

    if not is_mlx_installed():
        raise MLXInferenceError(
            "mlx-vlm is not installed. Please run 'Setup Apple Silicon' first."
        )

    model_folder = FASTVLM_MODELS.get(model_name)
    if not model_folder:
        raise MLXInferenceError(f"Unknown FastVLM model: {model_name}")

    mlx_model_path = MLX_CHECKPOINTS_DIR / model_folder
    if not mlx_model_path.exists():
        raise MLXInferenceError(
            f"MLX model {model_name} not found. Please run 'Setup Apple Silicon' first."
        )

    if progress_callback:
        progress_callback(f"Running MLX inference with {model_name}...")

    try:
        conda = _find_conda()
        if not conda:
            raise MLXInferenceError("Could not find conda.")

        # Run mlx_vlm.generate
        cmd = [
            conda, "run", "-n", CONDA_ENV_NAME,
            "python", "-m", "mlx_vlm.generate",
            "--model", str(mlx_model_path),
            "--image", str(image_path),
            "--prompt", prompt,
        ]

        logger.info(f"Running MLX inference: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        logger.debug(f"MLX stdout: {result.stdout}")
        logger.debug(f"MLX stderr: {result.stderr}")
        logger.debug(f"MLX returncode: {result.returncode}")

        if result.returncode != 0:
            logger.error(f"MLX inference failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise MLXInferenceError(f"MLX inference failed: {result.stderr}")

        # Parse output - mlx_vlm.generate prints the response
        output = result.stdout.strip()

        if not output:
            logger.error("MLX returned empty output")
            raise MLXInferenceError("MLX returned empty output.")

        logger.info(f"MLX inference successful, output length: {len(output)}")
        return output

    except subprocess.TimeoutExpired:
        logger.error("MLX inference timed out after 5 minutes")
        raise MLXInferenceError("MLX inference timed out after 5 minutes.")
    except MLXInferenceError:
        raise
    except Exception as e:
        logger.exception(f"MLX inference error: {str(e)}")
        raise MLXInferenceError(f"MLX inference error: {str(e)}")
