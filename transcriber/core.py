import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".mp3", ".wav", ".flac", ".ogg", ".m4a"}

BAR_WIDTH = 30
PAUSE_THRESHOLD = 1.5


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video file using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",                # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",       # 16kHz (optimal for Whisper)
        "-ac", "1",           # mono
        "-y",                 # overwrite
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_progress(current: float, total: float, elapsed: float) -> None:
    """Print a progress bar to stderr."""
    if total <= 0:
        return
    pct = min(current / total, 1.0)
    filled = int(BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    eta = (elapsed / pct - elapsed) if pct > 0.05 else 0
    eta_str = format_timestamp(eta) if eta > 0 else "--:--"
    print(f"\r  ┃{bar}┃ {pct:5.1%}  elapsed {format_timestamp(elapsed)}  eta {eta_str}", end="", flush=True)


def load_model(model_size: str = "medium") -> BatchedInferencePipeline:
    """Load and return a batched inference pipeline."""
    print(f"⏳ Loading model '{model_size}'...", end="", flush=True)
    t = time.monotonic()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    pipeline = BatchedInferencePipeline(model=model)
    print(f" done ({time.monotonic() - t:.1f}s)")
    return pipeline


def transcribe(input_path: Path, model_size: str = "medium", language: str | None = None, timestamps: bool = False, model: BatchedInferencePipeline | None = None, batch_size: int = 4) -> str:
    """Transcribe a video/audio file and return markdown content."""
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Unsupported format: {input_path.suffix}", file=sys.stderr)
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        sys.exit(1)

    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    is_audio = input_path.suffix.lower() in audio_extensions

    if model is None:
        model = load_model(model_size)

    # Step 1: Extract audio
    if is_audio:
        audio_file = str(input_path)
        tmp_dir = None
    else:
        tmp_dir = tempfile.mkdtemp()
        audio_file = str(Path(tmp_dir) / "audio.wav")
        print("  📦 Extracting audio...", end="", flush=True)
        t = time.monotonic()
        extract_audio(input_path, Path(audio_file))
        print(f" done ({time.monotonic() - t:.1f}s)")

    # Step 2: Transcribe with progress
    print("  🔊 Transcribing...")
    t_start = time.monotonic()
    segments, info = model.transcribe(
        audio_file,
        language=language,
        batch_size=batch_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    duration = info.duration

    paragraphs: list[str] = []
    current_parts: list[str] = []
    current_ts: float = 0.0
    prev_end: float = 0.0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        print_progress(segment.end, duration, time.monotonic() - t_start)

        gap = segment.start - prev_end if prev_end > 0 else 0.0
        ends_sentence = current_parts and current_parts[-1][-1:] in ".!?"

        if current_parts and (gap > PAUSE_THRESHOLD or ends_sentence):
            paragraphs.append((" ".join(current_parts), current_ts))
            current_parts = []
            current_ts = segment.start

        if not current_parts:
            current_ts = segment.start
        current_parts.append(text)
        prev_end = segment.end

    if current_parts:
        paragraphs.append((" ".join(current_parts), current_ts))

    elapsed = time.monotonic() - t_start
    print_progress(duration, duration, elapsed)
    print(f"\n  ✅ Done in {format_timestamp(elapsed)} | {info.language} ({info.language_probability:.0%}) | {format_timestamp(duration)} audio")

    # Step 3: Build markdown
    lines: list[str] = []
    lines.append(f"# Transcription: {input_path.name}\n")
    lines.append(f"- **Language:** {info.language} ({info.language_probability:.0%} confidence)")
    lines.append(f"- **Duration:** {format_timestamp(duration)}")
    lines.append(f"- **Model:** whisper-{model_size}")
    lines.append("")
    lines.append("## Content\n")

    for text, ts in paragraphs:
        if timestamps:
            lines.append(f"**[{format_timestamp(ts)}]** {text}\n")
        else:
            lines.append(f"{text}\n")

    # Cleanup temp audio
    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return "\n".join(lines)
