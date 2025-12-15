#!/bin/bash
# Build script for WTF Am I Doing?
# Creates a standalone .app bundle using PyInstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="WTF Am I Doing"
BUNDLE_ID="com.wtfamidoing.app"
PYTHON_PATH="/usr/local/opt/python@3.10/bin/python3.10"

echo "========================================="
echo "  Building $APP_NAME"
echo "========================================="
echo

# Check for Python with tkinter
if [ ! -x "$PYTHON_PATH" ]; then
    echo "Looking for Python 3.10+ with tkinter..."
    for py in /usr/local/opt/python@3.*/bin/python3.* /opt/homebrew/opt/python@3.*/bin/python3.*; do
        if [ -x "$py" ] && [[ ! "$py" == *-config ]] && $py -c "import tkinter" 2>/dev/null; then
            PYTHON_PATH="$py"
            break
        fi
    done
fi

if [ ! -x "$PYTHON_PATH" ]; then
    echo "ERROR: Could not find Python with tkinter support."
    echo "Install with: brew install python-tk@3.10"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist/*.app dist/"$APP_NAME"

# Create/update virtual environment
echo "Setting up virtual environment..."
if [ ! -d ".venv" ] || ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
    rm -rf .venv
    uv venv --python "$PYTHON_PATH"
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -e . pyinstaller --quiet

# Create app_launcher.py if it doesn't exist
if [ ! -f "app_launcher.py" ]; then
    echo "Creating app_launcher.py..."
    cat > app_launcher.py << 'EOF'
#!/usr/bin/env python3
"""Entry point for PyInstaller bundle."""

import sys
import os

# Add the src directory to path for imports
if getattr(sys, 'frozen', False):
    # Running as compiled
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)
else:
    # Running as script
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(bundle_dir, 'src'))

from wtf_am_i_doing.main import main

if __name__ == '__main__':
    main()
EOF
fi

# Build with PyInstaller
echo "Building app bundle with PyInstaller..."
.venv/bin/pyinstaller \
    --name "$APP_NAME" \
    --windowed \
    --onedir \
    --hidden-import tkinter \
    --hidden-import PIL \
    --hidden-import mss \
    --collect-submodules wtf_am_i_doing \
    --osx-bundle-identifier "$BUNDLE_ID" \
    --noconfirm \
    app_launcher.py

# Add screen recording permission to Info.plist
echo "Adding screen recording permission..."
/usr/libexec/PlistBuddy -c "Add :NSScreenCaptureUsageDescription string 'WTF Am I Doing needs screen recording permission to capture screenshots and describe your activity.'" "dist/$APP_NAME.app/Contents/Info.plist" 2>/dev/null || true

# Re-sign the app
echo "Signing app bundle..."
codesign --force --deep --sign - "dist/$APP_NAME.app"

echo
echo "========================================="
echo "  Build complete!"
echo "========================================="
echo
echo "App location: dist/$APP_NAME.app"
echo
echo "To install, drag to /Applications or run:"
echo "  open \"dist/$APP_NAME.app\""
echo
