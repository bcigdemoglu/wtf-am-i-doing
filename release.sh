#!/bin/bash
# Release script for WTF Am I Doing?
# Usage: ./release.sh <version>
# Example: ./release.sh 1.0.0

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 1.0.0"
    exit 1
fi

# Validate version format (basic check)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 1.0.0)"
    exit 1
fi

TAG="v$VERSION"
APP_NAME="WTF Am I Doing"
ZIP_NAME="WTF-Am-I-Doing-$VERSION.zip"

echo "=== Building $APP_NAME $TAG ==="
echo

# Change to script directory
cd "$(dirname "$0")"

# Check if tag already exists
if git tag -l "$TAG" | grep -q "$TAG"; then
    echo "Error: Tag $TAG already exists"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Run the build
echo "Running build..."
./build.sh

# Create zip
echo
echo "Creating release zip..."
cd dist
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "$APP_NAME.app"
cd ..

echo
echo "Zip created: dist/$ZIP_NAME"
ls -lh "dist/$ZIP_NAME"

# Create git tag
echo
echo "Creating git tag $TAG..."
git tag -a "$TAG" -m "Release $TAG"
git push origin "$TAG"

# Create GitHub release
echo
echo "Creating GitHub release..."
gh release create "$TAG" \
    --title "$APP_NAME $TAG" \
    --notes "## $APP_NAME $TAG

### Installation
1. Download \`$ZIP_NAME\`
2. Unzip and move \`$APP_NAME.app\` to Applications
3. On first run, grant Screen Recording permission when prompted

### Requirements
- macOS
- For FastVLM: conda (\`brew install --cask miniconda\`)
- For Claude: Claude CLI (\`npm install -g @anthropic-ai/claude-code\`)" \
    "dist/$ZIP_NAME"

echo
echo "=== Release $TAG complete! ==="
echo "https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$TAG"
