#!/bin/bash
# WTF Am I Doing? - Launcher Script
# Double-click this file to launch the application

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  WTF Am I Doing? - Launcher"
echo "========================================="
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed."
    echo
    read -p "Would you like to install uv? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Source the shell config to get uv in PATH
        if [ -f "$HOME/.zshrc" ]; then
            source "$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc"
        fi

        # Also try adding to PATH directly
        export PATH="$HOME/.local/bin:$PATH"
        export PATH="$HOME/.cargo/bin:$PATH"

        echo "uv installed successfully!"
    else
        echo "Cannot continue without uv. Exiting."
        exit 1
    fi
fi

echo "Using uv: $(which uv)"
echo

# Find a Python with tkinter support
find_python_with_tkinter() {
    # Try Python 3.10 from homebrew (known to have tkinter on this system)
    if [ -x "/usr/local/opt/python@3.10/bin/python3.10" ]; then
        if /usr/local/opt/python@3.10/bin/python3.10 -c "import tkinter" 2>/dev/null; then
            echo "/usr/local/opt/python@3.10/bin/python3.10"
            return
        fi
    fi
    # Try other homebrew Pythons
    for py in /usr/local/opt/python@3.*/bin/python3.*; do
        if [ -x "$py" ] && [[ ! "$py" == *-config ]]; then
            if $py -c "import tkinter" 2>/dev/null; then
                echo "$py"
                return
            fi
        fi
    done
    # Fall back to system Python
    if /usr/bin/python3 -c "import tkinter" 2>/dev/null; then
        echo "/usr/bin/python3"
        return
    fi
    echo ""
}

PYTHON_PATH=$(find_python_with_tkinter)
if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: Could not find a Python with tkinter support."
    echo "Please install python-tk: brew install python-tk@3.10"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"

# Remove existing venv if it doesn't have tkinter
if [ -d ".venv" ]; then
    if ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
        echo "Existing venv missing tkinter, recreating..."
        rm -rf .venv
    fi
fi

# Create venv with Python that has tkinter
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python "$PYTHON_PATH"
fi

# Install/update the package
echo "Installing wtf-am-i-doing..."
uv pip install -e . --quiet 2>/dev/null || uv pip install -e .
echo

# Run the application
echo "Starting application..."
echo
.venv/bin/python -m wtf_am_i_doing.main
