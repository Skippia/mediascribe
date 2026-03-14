#!/usr/bin/env bash
#
# Transcribe video/audio files to markdown.
# Uses uv to auto-manage the Python environment.
#
# Usage:
#   ./transcribe.sh video.mp4                  # single file
#   ./transcribe.sh ./lectures/                # all videos in folder
#   ./transcribe.sh video.mp4 -m small -l en
#   ./transcribe.sh video.mp4 -o output.md --timestamps
#   ./transcribe.sh --audio video.mp4            # extract .mp3 from video
#   ./transcribe.sh --audio ./lectures/         # extract .mp3 from all videos (recursive)
#   ./transcribe.sh video.mp4 --cloud           # cloud transcription via OpenRouter
#   ./transcribe.sh ./lectures/ --cloud --concurrency 3
#   ./transcribe.sh ./lectures/ --cloud --dry   # estimate cost without transcribing
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# Check dependencies
if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg is required but not installed." >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Error: uv is required but not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: transcribe.sh <file-or-folder> [options]"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE        Output markdown file (ignored for folders)"
    echo "  -m, --model MODEL        tiny|base|small|medium|large-v3 (default: medium)"
    echo "  -l, --language LANG      Language code: en, uk, ru, etc. (default: auto)"
    echo "      --timestamps         Include timestamps in output"
    echo "      --audio PATH         Extract .mp3 from video file or folder (recursive)"
    echo ""
    echo "Cloud transcription (OpenRouter):"
    echo "      --cloud              Use cloud API instead of local Whisper"
    echo "      --cloud-model MODEL  Cloud model (default: google/gemini-3-flash-preview)"
    echo "      --concurrency N      Parallel cloud jobs (default: 5)"
    echo "      --dry                Estimate cost without transcribing (requires --cloud)"
    exit 1
fi

cd "$SCRIPT_DIR"
exec uv run python -m transcriber.cli "$@"
