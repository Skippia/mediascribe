# transcribe-offline

> *Point at a video. Get a markdown transcript.*

Local video/audio transcription to clean markdown files. Pass a file or an entire folder — get accurate transcriptions with automatic language detection, paragraph merging, and a live progress bar. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend) running entirely on your machine. No API keys, no cloud, no costs.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![faster-whisper](https://img.shields.io/badge/Whisper-faster--whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)

## Why video-transcriber?

Uploading videos to Gemini/ChatGPT for transcription is slow, manual, and uncomfortable. This tool does it in one command — point at a file or folder, get `.md` files next to each video. Batch processing, smart paragraph merging, and it remembers what's already been transcribed.

## Features

- **Single file or entire folder** — pass a file path or a directory, it handles both
- **Smart paragraph merging** — Whisper segments are joined into coherent sentences based on punctuation and pauses, not dumped line-by-line
- **Skip already transcribed** — re-run on a folder safely; files with existing `.md` are skipped
- **Live progress bar** — percentage, elapsed time, and ETA for each file
- **Auto language detection** — or force a specific language with `-l`
- **Zero setup** — `uv` auto-creates the Python environment and installs dependencies on first run
- **100% local** — nothing leaves your machine

## How It Works

```
┌──────────┐    ┌──────────┐    ┌─────────────────┐    ┌──────────┐
│ Video    │───>│ ffmpeg   │───>│ faster-whisper   │───>│ Markdown │
│ (.mp4)   │    │ (audio)  │    │ (CTranslate2)   │    │ (.md)    │
└──────────┘    └──────────┘    └─────────────────┘    └──────────┘
                  16kHz mono        int8 quantized        paragraphs
                  PCM wav           CPU inference          merged
```

**Folder mode:**

```
┌────────────────┐    ┌─────────────┐    ┌──────────────────────┐
│ Scan folder    │───>│ Load model  │───>│ Transcribe each file │
│ for media      │    │ (once)      │    │ [1/N] ... [N/N]      │
│ files          │    │             │    │ skip if .md exists   │
└────────────────┘    └─────────────┘    └──────────────────────┘
```

## Supported Formats

| Video | Audio |
|-------|-------|
| `.mp4` `.mkv` `.webm` `.avi` `.mov` `.flv` `.wmv` | `.mp3` `.wav` `.flac` `.ogg` `.m4a` |

## Requirements

- **Linux** (tested on Arch/CachyOS)
- **Python 3.11+**
- **ffmpeg** — for audio extraction from video files
- **[uv](https://github.com/astral-sh/uv)** — Python package manager (auto-manages venv and dependencies)

## Installation

```bash
# Clone the repo
git clone https://github.com/Skippia/transcribe-offline.git
cd video-transcriber

# Make globally available
ln -s "$(pwd)/transcribe.sh" ~/.local/bin/transcribe

# That's it. First run will auto-install Python dependencies via uv.
```

### Install system dependencies (Arch Linux)

```bash
sudo pacman -S python ffmpeg
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

### Single file

```bash
transcribe video.mp4
# => creates video.md next to the video
```

### Entire folder

```bash
transcribe ./lectures/
# => creates .md for every media file in the folder
# => skips files that already have a .md
```

### With options

```bash
# Force English, use small model (faster)
transcribe video.mp4 -m small -l en

# Custom output path
transcribe video.mp4 -o notes.md

# Include timestamps
transcribe video.mp4 --timestamps
```

> **Note:** paths with spaces must be quoted: `transcribe "/path/to/my videos/"`

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Path to video/audio file or folder | *(required)* |
| `-o, --output` | Output markdown file (ignored for folders) | `<input>.md` |
| `-m, --model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` | `medium` |
| `-l, --language` | Language code: `en`, `uk`, `ru`, `de`, etc. | auto-detect |
| `--timestamps` | Include `[MM:SS]` timestamps in output | off |

## Model Size Guide

All models run locally on CPU with int8 quantization.

| Model | Speed (per 10 min audio) | Accuracy | RAM |
|-------|--------------------------|----------|-----|
| `tiny` | ~1 min | Low | ~0.5 GB |
| `base` | ~3 min | Good | ~0.5 GB |
| `small` | ~8 min | Better | ~1 GB |
| `medium` | ~15 min | Best practical | ~1.5 GB |
| `large-v3` | ~40 min | Maximum | ~3 GB |

`medium` is the default — best accuracy-to-speed ratio for CPU. Use `small` or `base` if you need faster processing and can tolerate lower accuracy.

## Output Example

```markdown
# Transcription: lecture-01.mp4

- **Language:** en (100% confidence)
- **Duration:** 07:25
- **Model:** whisper-medium

## Content

So foreplay and dirty talk, this is where it's at.

This is where the rubber hits the road because what happens, she starts to get extremely turned on.

She wants to get to a point where she wants you, where she can't wait.
```

## Progress Output

Single file:
```
⏳ Loading model 'medium'... done (3.1s)

📄 lecture-01.mp4
  📦 Extracting audio... done (0.2s)
  🔊 Transcribing...
  ┃████████████████░░░░░░░░░░░░░░┃ 54.2%  elapsed 00:26  eta 00:22
  ✅ Done in 00:35 | en (100%) | 01:04 audio
  💾 Saved: lecture-01.md

──────────────────────────────────────────────────
✅ All done in 00:35
```

Folder:
```
⏳ Loading model 'medium'... done (3.1s)

📂 ./lectures
   5 media file(s) found

──────────────────────────────────────────────────

🎬 [1/5] lecture-01.mp4
  📦 Extracting audio... done (0.2s)
  🔊 Transcribing...
  ┃██████████████████████████████┃ 100.0%  elapsed 00:35  eta --:--
  ✅ Done in 00:35 | en (100%) | 01:04 audio
  💾 Saved: lecture-01.md

⏭️  [2/5] lecture-02.mp4 — skipped (already transcribed)

🎬 [3/5] lecture-03.mp4
  ...

──────────────────────────────────────────────────
✅ All done in 04:26
   📊 4 transcribed, 1 skipped, 5 total
```

## File Structure

```
transcribe-offline/
├── transcribe.sh          # Shell entry point (symlink this to ~/.local/bin/)
├── pyproject.toml         # Python project config & dependencies
├── transcriber/
│   ├── __init__.py
│   ├── cli.py             # Argument parsing, file/folder dispatch
│   └── core.py            # Audio extraction, Whisper inference, markdown output
└── README.md
```

## How It Works Under the Hood

1. **Audio extraction** — `ffmpeg` converts video to 16kHz mono WAV (optimal for Whisper)
2. **Inference** — `faster-whisper` runs the Whisper model with int8 quantization via CTranslate2
3. **Paragraph merging** — raw Whisper segments (2-5 seconds each) are merged into coherent paragraphs based on sentence-ending punctuation and pauses >1.5 seconds
4. **Markdown output** — clean markdown with metadata header and content body

In folder mode, the model is loaded **once** and reused for all files (model loading takes ~3s, so this saves significant time on batch jobs).

## License

MIT
