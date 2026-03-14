import base64
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from transcriber.core import SUPPORTED_EXTENSIONS, format_timestamp

DEFAULT_MODEL = "google/gemini-3-flash-preview"

AUDIO_TOKEN_RATE = 32  # Gemini: 32 tokens per second of audio

# Audio input pricing per 1M tokens (USD)
MODEL_AUDIO_PRICING: dict[str, float] = {
    "google/gemini-3-flash-preview": 1.00,
    "google/gemini-2.5-flash": 1.00,
    "google/gemini-2.5-pro": 1.25,
    "google/gemini-2.0-flash-001": 0.70,
}

TRANSCRIPTION_PROMPT = """\
Transcribe the following audio into text. Return ONLY a JSON array of objects, each with:
- "time": timestamp in "HH:MM:SS" format (approximate start time of the paragraph)
- "text": the transcribed text for that paragraph

Group the text into natural paragraphs (every few sentences or at topic changes).
Do not include any other text, markdown formatting, or code fences — just the raw JSON array.

{lang_instruction}

Example output format:
[
  {{"time": "00:00:00", "text": "Welcome to today's lecture. We'll be covering..."}},
  {{"time": "00:02:15", "text": "The first topic is..."}}
]"""


@dataclass
class CloudResult:
    paragraphs: list[tuple[str, float]]
    language: str
    duration: float
    model: str


def _compress_audio(input_path: Path, output_path: Path) -> None:
    """Compress audio/video to 32kbps mono MP3 for minimal upload size."""
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-vn",                    # no video
        "-acodec", "libmp3lame",
        "-b:a", "32k",            # 32kbps — sufficient for speech
        "-ac", "1",               # mono
        "-ar", "16000",           # 16kHz
        "-y",                     # overwrite
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to compress audio: {input_path}")


def get_duration(path: Path) -> float:
    """Get audio/video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def estimate_cost(duration_secs: float, model: str = DEFAULT_MODEL) -> tuple[int, float]:
    """Estimate audio tokens and USD cost for a given duration.

    Returns (audio_tokens, estimated_cost_usd).
    Includes a rough estimate for output tokens (~200 tokens/min of audio).
    """
    audio_tokens = int(duration_secs * AUDIO_TOKEN_RATE)
    output_tokens = int((duration_secs / 60) * 200)

    price_per_m = MODEL_AUDIO_PRICING.get(model, 1.00)
    audio_cost = (audio_tokens / 1_000_000) * price_per_m
    # Output pricing: ~$0.40/M for Flash models, use as rough estimate
    output_cost = (output_tokens / 1_000_000) * 0.40
    return audio_tokens + output_tokens, audio_cost + output_cost


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


def _parse_response(text: str) -> list[tuple[str, float]]:
    """Parse the JSON response from the LLM into (text, timestamp) tuples."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Failed to parse JSON response: {e}", file=sys.stderr)
        print(f"  Raw response (first 500 chars): {text[:500]}", file=sys.stderr)
        # Fallback: return the whole response as a single paragraph
        return [(text.strip(), 0.0)]

    paragraphs = []
    for item in data:
        ts = _parse_timestamp(item.get("time", "00:00:00"))
        para_text = item.get("text", "").strip()
        if para_text:
            paragraphs.append((para_text, ts))

    return paragraphs


def transcribe_cloud(
    input_path: Path,
    language: str | None,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> CloudResult:
    """Transcribe an audio/video file via OpenRouter cloud API."""
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Unsupported format: {input_path.suffix}", file=sys.stderr)
        raise ValueError(f"Unsupported format: {input_path.suffix}")

    duration = get_duration(input_path)

    # Compress to 32kbps MP3
    print("  📦 Compressing audio...", end="", flush=True)
    t = time.monotonic()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_mp3 = Path(tmp.name)

    try:
        _compress_audio(input_path, tmp_mp3)
        mp3_size = tmp_mp3.stat().st_size / (1024 * 1024)
        print(f" done ({time.monotonic() - t:.1f}s, {mp3_size:.1f}MB)")

        # Base64-encode
        audio_b64 = base64.standard_b64encode(tmp_mp3.read_bytes()).decode("ascii")
    finally:
        tmp_mp3.unlink(missing_ok=True)

    # Build prompt
    lang_instruction = ""
    if language:
        lang_instruction = f"The audio is in language code '{language}'. Transcribe in that language."

    prompt = TRANSCRIPTION_PROMPT.format(lang_instruction=lang_instruction)

    # Call OpenRouter API
    print("  ☁️  Sending to cloud API...", end="", flush=True)
    t = time.monotonic()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "mp3",
                        },
                    },
                ],
            }
        ],
    )

    api_elapsed = time.monotonic() - t
    print(f" done ({api_elapsed:.1f}s)")

    # Parse response
    reply = response.choices[0].message.content or ""
    paragraphs = _parse_response(reply)

    # Detect language from response if not specified
    detected_lang = language or "auto"

    usage = response.usage
    if usage:
        cost_in = (usage.prompt_tokens / 1_000_000) * 0.1   # Gemini Flash input
        cost_out = (usage.completion_tokens / 1_000_000) * 0.4  # Gemini Flash output
        print(f"  💰 Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out (~${cost_in + cost_out:.4f})")

    print(f"  ✅ Done | {len(paragraphs)} paragraphs | {format_timestamp(duration)} audio")

    return CloudResult(
        paragraphs=paragraphs,
        language=detected_lang,
        duration=duration,
        model=model,
    )
