# Weighted Temporal Forecaster (WTF) Am I Doing?

A macOS app that periodically captures your screen and uses AI to describe what you're doing. Creates a searchable activity diary in JSON format.

## Features

- **Automatic screen capture** at configurable intervals (5s - 5min)
- **Multiple AI backends**:
  - FastVLM (local, runs on Apple Silicon/Intel via conda)
  - Claude (via Claude CLI - opus, sonnet, haiku)
- **Activity diary** saved as JSON with timestamps and inference times
- **Clickable preview** - click thumbnail to open full screenshot
- **Settings persistence** - remembers your preferences between sessions
- **Resolution options** - High, Medium, Low to balance quality vs speed

## Requirements

- macOS
- Python 3.10+ with tkinter (`brew install python-tk@3.10`)
- For FastVLM: conda (`brew install --cask miniconda`)
- For Claude: Claude CLI

## Build

```bash
./build.sh
```

## Release

```bash
./release.sh 1.0.0
```

Creates a GitHub release with the built `.app` bundle.

## Run

```bash
open "dist/WTF Am I Doing.app"
```

On first run, grant **Screen Recording** permission when prompted (System Preferences > Privacy & Security > Screen Recording).

## Usage

1. Select a model from the dropdown
2. For FastVLM models, click "Download FastVLM" first (one-time setup)
3. Adjust interval and resolution as needed
4. Click **Play** to start capturing
5. Click **Open Diary** to view your activity log

## Data Location

All data is stored in `~/.wtf-am-i-doing/`:
- `diary.json` - Activity entries
- `error.json` - Error log
- `settings_cached.json` - Your preferences
- `app.log` - Debug log

## License

MIT
