# WTF Am I Doing?

**Weighted Temporal Forecaster** - A continuous screen activity diary using FastVLM.

## What is this?

A macOS application that:
1. Takes screenshots of all your monitors periodically
2. Runs them through Apple's FastVLM to understand what you're doing
3. Saves a continuous diary of your activity in JSON format

## Quick Start

1. Double-click `launch.command` to start the app
2. Click "Download Models" to set up FastVLM (first time only)
3. Click "Play" to start recording

## Requirements

- macOS (Intel or Apple Silicon)
- Python 3.10+
- GitHub CLI (`gh`) for model download: `brew install gh`

## Manual Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv venv
uv pip install -e .

# Run
uv run python -m wtf_am_i_doing.main
```

## Diary Format

Entries are saved to `~/.wtf-am-i-doing/diary.json`:

```json
{
  "entries": [
    {
      "timestamp": "2025-12-10T14:32:15.123Z",
      "model": "fastvlm_0.5b_stage3",
      "description": "User is writing Python code in VS Code..."
    }
  ]
}
```

## License

MIT
