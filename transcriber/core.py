import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".aac", ".opus", ".wma", ".aiff", ".aif",
    ".amr", ".ape", ".ac3", ".dts", ".mka",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".divx",
    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".3gp", ".m4v",
    ".vob", ".ogv", ".asf",
}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

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


def extract_audio_mp3(input_path: Path, output_path: Path) -> bool:
    """Extract audio from a video file as high-quality MP3. Returns True on success."""
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-vn",                 # no video
        "-acodec", "libmp3lame",
        "-q:a", "2",           # VBR ~190kbps (high quality)
        "-y",                  # overwrite
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()
        return False
    return True


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


def build_paragraphs(segments, pause_threshold: float = PAUSE_THRESHOLD) -> list[tuple[str, float]]:
    """Group segments into paragraphs based on pauses and sentence endings.

    Each segment must have .start, .end, and .text attributes.
    Returns list of (paragraph_text, start_timestamp) tuples.
    """
    paragraphs: list[tuple[str, float]] = []
    current_parts: list[str] = []
    current_ts: float = 0.0
    prev_end: float = 0.0

    for segment in segments:
        text = segment.text.strip() if hasattr(segment, "text") else str(segment.get("text", "")).strip()
        start = segment.start if hasattr(segment, "start") else segment["start"]
        end = segment.end if hasattr(segment, "end") else segment["end"]

        if not text:
            continue

        gap = start - prev_end if prev_end > 0 else 0.0
        ends_sentence = current_parts and current_parts[-1][-1:] in ".!?"

        if current_parts and (gap > pause_threshold or ends_sentence):
            paragraphs.append((" ".join(current_parts), current_ts))
            current_parts = []
            current_ts = start

        if not current_parts:
            current_ts = start
        current_parts.append(text)
        prev_end = end

    if current_parts:
        paragraphs.append((" ".join(current_parts), current_ts))

    return paragraphs


def build_markdown(
    input_name: str,
    paragraphs: list[tuple[str, float]],
    timestamps: bool,
    language: str,
    lang_prob: float,
    duration: float,
    model_name: str,
) -> str:
    """Build markdown output from paragraphs and metadata."""
    lines: list[str] = []
    lines.append(f"# Transcription: {input_name}\n")
    lines.append(f"- **Language:** {language} ({lang_prob:.0%} confidence)")
    lines.append(f"- **Duration:** {format_timestamp(duration)}")
    lines.append(f"- **Model:** {model_name}")
    lines.append("")
    lines.append("## Content\n")

    for text, ts in paragraphs:
        if timestamps:
            lines.append(f"**[{format_timestamp(ts)}]** {text}\n")
        else:
            lines.append(f"{text}\n")

    return "\n".join(lines)


def transcribe(input_path: Path, model_size: str = "medium", language: str | None = None, timestamps: bool = False, model: BatchedInferencePipeline | None = None, batch_size: int = 4) -> str:
    """Transcribe a video/audio file and return markdown content."""
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Unsupported format: {input_path.suffix}", file=sys.stderr)
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        sys.exit(1)

    is_audio = input_path.suffix.lower() in AUDIO_EXTENSIONS

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

    all_segments = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        print_progress(segment.end, duration, time.monotonic() - t_start)
        all_segments.append(segment)

    paragraphs = build_paragraphs(all_segments)

    elapsed = time.monotonic() - t_start
    print_progress(duration, duration, elapsed)
    print(f"\n  ✅ Done in {format_timestamp(elapsed)} | {info.language} ({info.language_probability:.0%}) | {format_timestamp(duration)} audio")

    md = build_markdown(
        input_name=input_path.name,
        paragraphs=paragraphs,
        timestamps=timestamps,
        language=info.language,
        lang_prob=info.language_probability,
        duration=duration,
        model_name=f"whisper-{model_size}",
    )

    # Cleanup temp audio
    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return md
