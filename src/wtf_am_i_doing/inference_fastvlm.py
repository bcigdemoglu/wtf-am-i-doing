"""FastVLM inference backend."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from .config import (
    APP_DIR,
    CHECKPOINTS_DIR,
    FASTVLM_DIR,
    FASTVLM_MODELS,
    is_fastvlm_cloned,
)

logger = logging.getLogger(__name__)

# URL to download FastVLM as zip
FASTVLM_ZIP_URL = "https://github.com/apple/ml-fastvlm/archive/refs/heads/main.zip"

CONDA_ENV_NAME = "fastvlm"


class FastVLMSetupError(Exception):
    """Raised when FastVLM setup/download fails."""
    pass


class FastVLMInferenceError(Exception):
    """Raised when FastVLM inference fails."""
    pass


def _find_conda() -> str | None:
    """Find the conda executable, checking common locations."""
    logger.debug("Looking for conda executable...")

    # First try PATH
    conda_path = shutil.which("conda")
    if conda_path:
        logger.debug(f"Found conda in PATH: {conda_path}")
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
        logger.debug(f"Checking: {p}")
        if p.exists():
            logger.info(f"Found conda at: {p}")
            return str(p)

    logger.error("Could not find conda in PATH or common locations")
    return None


def _is_conda_installed() -> bool:
    """Check if conda is installed."""
    return _find_conda() is not None


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


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    import platform
    return platform.machine() == "arm64"


def _patch_fastvlm_requirements() -> None:
    """
    Patch FastVLM requirements for Intel Mac compatibility.
    PyTorch 2.6.0 doesn't support Intel Macs, so we use 2.2.0.
    """
    if _is_apple_silicon():
        return  # No patch needed for Apple Silicon

    pyproject_path = FASTVLM_DIR / "pyproject.toml"
    if not pyproject_path.exists():
        return

    content = pyproject_path.read_text()

    # Replace torch and torchvision versions for Intel Mac compatibility
    patches = [
        ('torch==2.6.0', 'torch>=2.0.0,<2.3.0'),
        ('torchvision==0.21.0', 'torchvision>=0.15.0,<0.18.0'),
    ]

    modified = False
    for old, new in patches:
        if old in content:
            content = content.replace(old, new)
            modified = True

    if modified:
        pyproject_path.write_text(content)


def clone_fastvlm(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Download the FastVLM repository as a zip file.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        FastVLMSetupError: If download fails
    """
    if is_fastvlm_cloned():
        if progress_callback:
            progress_callback("FastVLM already downloaded.")
        return

    # Ensure app directory exists
    APP_DIR.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Downloading FastVLM repository...")

    try:
        # Download zip file to temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "fastvlm.zip"
            extract_dir = Path(tmpdir) / "extract"

            # Download using curl
            result = subprocess.run(
                ["curl", "-L", "-o", str(zip_path), FASTVLM_ZIP_URL],
                capture_output=True,
                text=True,
                check=True,
            )

            if progress_callback:
                progress_callback("Extracting FastVLM...")

            # Extract zip
            result = subprocess.run(
                ["unzip", "-q", str(zip_path), "-d", str(extract_dir)],
                capture_output=True,
                text=True,
                check=True,
            )

            # The zip extracts to ml-fastvlm-main/, move it to FASTVLM_DIR
            extracted_folder = extract_dir / "ml-fastvlm-main"
            if not extracted_folder.exists():
                # Try to find the extracted folder
                dirs = list(extract_dir.iterdir())
                if dirs:
                    extracted_folder = dirs[0]
                else:
                    raise FastVLMSetupError("Failed to find extracted FastVLM folder")

            # Remove existing directory if it exists
            if FASTVLM_DIR.exists():
                shutil.rmtree(FASTVLM_DIR)

            # Move to final location
            shutil.move(str(extracted_folder), str(FASTVLM_DIR))

        if progress_callback:
            progress_callback("FastVLM downloaded successfully.")
    except subprocess.CalledProcessError as e:
        raise FastVLMSetupError(f"Failed to download FastVLM: {e.stderr}")
    except Exception as e:
        raise FastVLMSetupError(f"Failed to download FastVLM: {str(e)}")


def install_fastvlm_deps(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Install FastVLM dependencies using conda (as recommended by FastVLM).

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        FastVLMSetupError: If installation fails
    """
    if not is_fastvlm_cloned():
        raise FastVLMSetupError("FastVLM is not cloned. Clone it first.")

    if not _is_conda_installed():
        raise FastVLMSetupError(
            "Conda is not installed.\n\n"
            "Install Miniconda with:\n"
            "  brew install --cask miniconda\n\n"
            "Then restart the app."
        )

    try:
        # Patch requirements for Intel Mac if needed
        if not _is_apple_silicon():
            if progress_callback:
                progress_callback("Patching requirements for Intel Mac...")
            _patch_fastvlm_requirements()

        conda = _find_conda()
        if not conda:
            raise FastVLMSetupError("Could not find conda executable. Please install miniconda.")

        logger.info(f"Using conda at: {conda}")

        # Create conda environment if it doesn't exist
        if not _conda_env_exists():
            if progress_callback:
                progress_callback("Creating conda environment (python=3.10)...")

            logger.info(f"Creating conda env: {CONDA_ENV_NAME}")
            result = subprocess.run(
                [conda, "create", "-n", CONDA_ENV_NAME, "python=3.10", "-y"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"Conda create stdout: {result.stdout}")
            logger.debug(f"Conda create stderr: {result.stderr}")

        if progress_callback:
            progress_callback("Installing FastVLM dependencies (this may take a while)...")

        # Install dependencies using pip in the conda environment
        logger.info("Installing FastVLM dependencies via pip")
        result = subprocess.run(
            [conda, "run", "-n", CONDA_ENV_NAME, "pip", "install", "-e", "."],
            cwd=str(FASTVLM_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"Pip install stdout: {result.stdout}")
        logger.debug(f"Pip install stderr: {result.stderr}")

        if progress_callback:
            progress_callback("FastVLM dependencies installed.")
    except subprocess.CalledProcessError as e:
        raise FastVLMSetupError(f"Failed to install FastVLM dependencies: {e.stderr}")
    except Exception as e:
        raise FastVLMSetupError(f"Failed to install FastVLM dependencies: {str(e)}")


def download_models(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Download FastVLM model checkpoints.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        FastVLMSetupError: If download fails
    """
    if not is_fastvlm_cloned():
        raise FastVLMSetupError("FastVLM is not cloned. Clone it first.")

    if progress_callback:
        progress_callback("Downloading models (this will take a while, grab a coffee)...")

    get_models_script = FASTVLM_DIR / "get_models.sh"

    if not get_models_script.exists():
        raise FastVLMSetupError("get_models.sh not found in FastVLM directory.")

    try:
        result = subprocess.run(
            ["bash", str(get_models_script)],
            cwd=str(FASTVLM_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        if progress_callback:
            progress_callback("Models downloaded successfully.")
    except subprocess.CalledProcessError as e:
        raise FastVLMSetupError(f"Failed to download models: {e.stderr}")


def run_inference(
    image_path: Path,
    model_name: str,
    prompt: str,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """
    Run FastVLM inference on an image.

    Args:
        image_path: Path to the image file
        model_name: Display name of the model (e.g., "FastVLM-0.5B")
        prompt: Text prompt for the model
        progress_callback: Optional callback for progress updates

    Returns:
        The model's text output

    Raises:
        FastVLMInferenceError: If inference fails
    """
    if not is_fastvlm_cloned():
        raise FastVLMInferenceError("FastVLM is not set up. Please download models first.")

    model_folder = FASTVLM_MODELS.get(model_name)
    if not model_folder:
        raise FastVLMInferenceError(f"Unknown FastVLM model: {model_name}")

    model_path = CHECKPOINTS_DIR / model_folder
    if not model_path.exists():
        raise FastVLMInferenceError(
            f"Model {model_name} not found. Please download models first."
        )

    predict_script = FASTVLM_DIR / "predict.py"
    if not predict_script.exists():
        raise FastVLMInferenceError("predict.py not found in FastVLM directory.")

    if progress_callback:
        progress_callback(f"Running inference with {model_name}...")

    try:
        # Find conda
        conda = _find_conda()
        if not conda:
            raise FastVLMInferenceError(
                "Could not find conda. Please install miniconda."
            )
        logger.info(f"Using conda at: {conda}")

        # Check if conda environment exists
        if not _conda_env_exists():
            raise FastVLMInferenceError(
                "FastVLM conda environment not found. Please run 'Download Models' first."
            )

        # Run predict.py using conda run
        cmd = [
            conda, "run", "-n", CONDA_ENV_NAME,
            "python", str(predict_script),
            "--model-path", str(model_path),
            "--image-file", str(image_path),
            "--prompt", prompt,
        ]
        logger.info(f"Running inference command: {' '.join(cmd)}")
        logger.debug(f"Working directory: {FASTVLM_DIR}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(FASTVLM_DIR),
            timeout=300,  # 5 minute timeout
        )

        logger.debug(f"Inference stdout: {result.stdout}")
        logger.debug(f"Inference stderr: {result.stderr}")
        logger.debug(f"Inference returncode: {result.returncode}")

        if result.returncode != 0:
            logger.error(f"Inference failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise FastVLMInferenceError(f"Inference failed: {result.stderr}")

        # The output should be in stdout
        output = result.stdout.strip()

        if not output:
            logger.error("Model returned empty output")
            raise FastVLMInferenceError("Model returned empty output.")

        logger.info(f"Inference successful, output length: {len(output)}")
        return output

    except subprocess.TimeoutExpired:
        logger.error("Inference timed out after 5 minutes")
        raise FastVLMInferenceError("Inference timed out after 5 minutes.")
    except FastVLMInferenceError:
        raise
    except Exception as e:
        logger.exception(f"Inference error: {str(e)}")
        raise FastVLMInferenceError(f"Inference error: {str(e)}")


def full_setup(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Perform full FastVLM setup: clone repo, install deps, download models.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        FastVLMSetupError: If any step fails
    """
    clone_fastvlm(progress_callback)
    install_fastvlm_deps(progress_callback)
    download_models(progress_callback)

    if progress_callback:
        progress_callback("Setup complete! All models are ready.")
