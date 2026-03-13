import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".mp3", ".wav", ".flac", ".ogg", ".m4a"}


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


def transcribe(input_path: Path, model_size: str = "medium", language: str | None = None, timestamps: bool = False) -> str:
    """Transcribe a video/audio file and return markdown content."""
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Unsupported format: {input_path.suffix}", file=sys.stderr)
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        sys.exit(1)

    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    is_audio = input_path.suffix.lower() in audio_extensions

    print(f"Loading model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if is_audio:
        audio_file = str(input_path)
        tmp_dir = None
    else:
        tmp_dir = tempfile.mkdtemp()
        audio_file = str(Path(tmp_dir) / "audio.wav")
        print("Extracting audio...")
        extract_audio(input_path, Path(audio_file))

    print("Transcribing...")
    segments, info = model.transcribe(audio_file, language=language, beam_size=5)

    lines: list[str] = []
    lines.append(f"# Transcription: {input_path.name}\n")
    lines.append(f"- **Language:** {info.language} ({info.language_probability:.0%} confidence)")
    lines.append(f"- **Duration:** {format_timestamp(info.duration)}")
    lines.append(f"- **Model:** whisper-{model_size}")
    lines.append("")

    lines.append("## Content\n")

    # Collect all segments, then merge into coherent paragraphs.
    # A new paragraph starts when there's a pause > 1.5s between segments
    # or the previous segment ended with sentence-ending punctuation.
    PAUSE_THRESHOLD = 1.5
    paragraphs: list[str] = []
    current_parts: list[str] = []
    current_ts: float = 0.0
    prev_end: float = 0.0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

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
