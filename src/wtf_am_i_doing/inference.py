"""FastVLM inference wrapper."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .config import (
    APP_DIR,
    CHECKPOINTS_DIR,
    FASTVLM_DIR,
    MODELS,
    is_fastvlm_cloned,
)

# URL to download FastVLM as zip
FASTVLM_ZIP_URL = "https://github.com/apple/ml-fastvlm/archive/refs/heads/main.zip"


class InferenceError(Exception):
    """Raised when inference fails."""
    pass


class SetupError(Exception):
    """Raised when setup/download fails."""
    pass


def clone_fastvlm(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Download the FastVLM repository as a zip file.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        SetupError: If download fails
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
                    raise SetupError("Failed to find extracted FastVLM folder")

            # Remove existing directory if it exists
            if FASTVLM_DIR.exists():
                shutil.rmtree(FASTVLM_DIR)

            # Move to final location
            shutil.move(str(extracted_folder), str(FASTVLM_DIR))

        if progress_callback:
            progress_callback("FastVLM downloaded successfully.")
    except subprocess.CalledProcessError as e:
        raise SetupError(f"Failed to download FastVLM: {e.stderr}")
    except Exception as e:
        raise SetupError(f"Failed to download FastVLM: {str(e)}")


def get_fastvlm_python() -> Path:
    """Get the path to the FastVLM venv's Python executable."""
    return FASTVLM_DIR / ".venv" / "bin" / "python"


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


def install_fastvlm_deps(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Install FastVLM dependencies in a dedicated venv.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        SetupError: If installation fails
    """
    if not is_fastvlm_cloned():
        raise SetupError("FastVLM is not cloned. Clone it first.")

    venv_dir = FASTVLM_DIR / ".venv"

    try:
        # Patch requirements for Intel Mac if needed
        if not _is_apple_silicon():
            if progress_callback:
                progress_callback("Patching requirements for Intel Mac...")
            _patch_fastvlm_requirements()

        # Create venv if it doesn't exist
        if not venv_dir.exists():
            if progress_callback:
                progress_callback("Creating FastVLM virtual environment...")

            result = subprocess.run(
                ["uv", "venv", str(venv_dir)],
                cwd=str(FASTVLM_DIR),
                capture_output=True,
                text=True,
                check=True,
            )

        if progress_callback:
            progress_callback("Installing FastVLM dependencies (this may take a while)...")

        # Install dependencies into the venv
        result = subprocess.run(
            ["uv", "pip", "install", "-e", ".", "--python", str(get_fastvlm_python())],
            cwd=str(FASTVLM_DIR),
            capture_output=True,
            text=True,
            check=True,
        )

        if progress_callback:
            progress_callback("FastVLM dependencies installed.")
    except subprocess.CalledProcessError as e:
        raise SetupError(f"Failed to install FastVLM dependencies: {e.stderr}")
    except Exception as e:
        raise SetupError(f"Failed to install FastVLM dependencies: {str(e)}")


def download_models(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Download FastVLM model checkpoints.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        SetupError: If download fails
    """
    if not is_fastvlm_cloned():
        raise SetupError("FastVLM is not cloned. Clone it first.")

    if progress_callback:
        progress_callback("Downloading models (this will take a while, grab a coffee)...")

    get_models_script = FASTVLM_DIR / "get_models.sh"

    if not get_models_script.exists():
        raise SetupError("get_models.sh not found in FastVLM directory.")

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
        raise SetupError(f"Failed to download models: {e.stderr}")


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
        InferenceError: If inference fails
    """
    if not is_fastvlm_cloned():
        raise InferenceError("FastVLM is not set up. Please download models first.")

    model_folder = MODELS.get(model_name)
    if not model_folder:
        raise InferenceError(f"Unknown model: {model_name}")

    model_path = CHECKPOINTS_DIR / model_folder
    if not model_path.exists():
        raise InferenceError(
            f"Model {model_name} not found. Please download models first."
        )

    predict_script = FASTVLM_DIR / "predict.py"
    if not predict_script.exists():
        raise InferenceError("predict.py not found in FastVLM directory.")

    if progress_callback:
        progress_callback(f"Running inference with {model_name}...")

    try:
        # Use the FastVLM venv's Python
        fastvlm_python = get_fastvlm_python()
        if not fastvlm_python.exists():
            raise InferenceError(
                "FastVLM Python environment not found. Please run 'Download Models' first."
            )

        # Run predict.py with the image and prompt
        result = subprocess.run(
            [
                str(fastvlm_python),
                str(predict_script),
                "--model-path", str(model_path),
                "--image-file", str(image_path),
                "--prompt", prompt,
            ],
            capture_output=True,
            text=True,
            cwd=str(FASTVLM_DIR),
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise InferenceError(f"Inference failed: {result.stderr}")

        # The output should be in stdout
        output = result.stdout.strip()

        if not output:
            raise InferenceError("Model returned empty output.")

        return output

    except subprocess.TimeoutExpired:
        raise InferenceError("Inference timed out after 5 minutes.")
    except Exception as e:
        raise InferenceError(f"Inference error: {str(e)}")


def full_setup(progress_callback: Callable[[str], None] | None = None) -> None:
    """
    Perform full setup: clone repo, install deps, download models.

    Args:
        progress_callback: Optional callback for progress updates

    Raises:
        SetupError: If any step fails
    """
    clone_fastvlm(progress_callback)
    install_fastvlm_deps(progress_callback)
    download_models(progress_callback)

    if progress_callback:
        progress_callback("Setup complete! All models are ready.")
