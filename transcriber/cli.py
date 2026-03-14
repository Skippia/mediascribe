import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcriber.core import SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, build_markdown, extract_audio_mp3, format_timestamp, load_model, transcribe


def transcribe_file(input_path: Path, output_path: Path, model, language: str | None, timestamps: bool, batch_size: int) -> float:
    """Transcribe a single file. Returns elapsed time in seconds."""
    t = time.monotonic()
    result = transcribe(
        input_path=input_path.resolve(),
        model=model,
        language=language,
        timestamps=timestamps,
        batch_size=batch_size,
    )
    output_path.write_text(result, encoding="utf-8")
    print(f"  💾 Saved: {output_path.name}")
    return time.monotonic() - t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio files to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  transcribe video.mp4\n"
               "  transcribe ./lectures/              # all videos in folder\n"
               "  transcribe lecture.mp4 -m small -l en\n"
               "  transcribe podcast.mp3 -o notes.md --timestamps\n"
               "  transcribe --audio video.mp4        # extract .mp3 from video\n"
               "  transcribe --audio ./lectures/      # extract .mp3 from all videos (recursive)",
    )
    parser.add_argument("input", type=Path, nargs="?", help="Path to video/audio file or folder")
    parser.add_argument("-o", "--output", type=Path, help="Output file path (ignored for folders)")
    parser.add_argument("-m", "--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("-l", "--language", default=None, help="Language code, e.g. 'en', 'uk', 'ru' (default: auto-detect)")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in output (off by default)")
    parser.add_argument("-b", "--batch-size", type=int, default=4,
                        help="Batch size for inference — lower = less RAM (default: 4)")
    parser.add_argument("--audio", type=Path, metavar="PATH",
                        help="Extract .mp3 from video file or folder (recursive)")

    # Cloud transcription options
    parser.add_argument("--cloud", action="store_true",
                        help="Use cloud API (OpenRouter) instead of local Whisper")
    parser.add_argument("--cloud-model", default="google/gemini-3-flash-preview",
                        help="Cloud model to use (default: google/gemini-3-flash-preview)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of parallel cloud transcriptions (default: 5)")
    parser.add_argument("--dry", action="store_true",
                        help="Estimate cost without transcribing (requires --cloud)")

    args = parser.parse_args()

    if args.dry and not args.cloud:
        print("Error: --dry requires --cloud", file=sys.stderr)
        sys.exit(1)

    if args.audio:
        _run_audio_extraction(args.audio, args.output)
    elif args.input:
        if not args.input.exists():
            print(f"Error: path not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        if args.cloud:
            _run_cloud_transcription(args)
        else:
            _run_transcription(args)
    else:
        parser.print_help()
        sys.exit(1)


def _extract_single_audio(input_path: Path, output_path: Path) -> tuple[bool, float]:
    """Extract audio from a single video file. Returns (success, elapsed)."""
    t = time.monotonic()
    ok = extract_audio_mp3(input_path, output_path)
    elapsed = time.monotonic() - t
    if ok:
        print(f"  ✅ Done in {format_timestamp(elapsed)}")
        print(f"  💾 Saved: {output_path}")
    else:
        print(f"  ❌ Failed (ffmpeg error)")
    return ok, elapsed


def _run_audio_extraction(target: Path, output: Path | None) -> None:
    """Handle --audio mode: extract .mp3 from video file or folder (recursive)."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        sys.exit(1)

    if target.is_file():
        suffix = target.suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            print(f"Error: {target.name} is not a video file (got '{suffix}')", file=sys.stderr)
            sys.exit(1)

        output_path = output or target.with_suffix(".mp3")
        print(f"\n🎵 Extracting audio: {target.name}")
        ok, elapsed = _extract_single_audio(target, output_path)
        print(f"\n{'─' * 50}")
        if ok:
            print(f"✅ All done in {format_timestamp(elapsed)}")
        else:
            sys.exit(1)

    elif target.is_dir():
        files = sorted(
            f for f in target.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not files:
            print(f"No video files found in: {target}", file=sys.stderr)
            sys.exit(1)

        total = len(files)
        skipped = 0
        processed = 0
        failed = 0
        t_total = time.monotonic()

        print(f"\n📂 {target} (recursive)")
        print(f"   {total} video file(s) found\n")
        print(f"{'─' * 50}")

        for i, f in enumerate(files, 1):
            mp3_path = f.with_suffix(".mp3")
            if mp3_path.exists():
                skipped += 1
                print(f"\n⏭️  [{i}/{total}] {f.relative_to(target)} — skipped (.mp3 exists)")
                continue
            print(f"\n🎵 [{i}/{total}] {f.relative_to(target)}")
            ok, _ = _extract_single_audio(f, mp3_path)
            if ok:
                processed += 1
            else:
                failed += 1

        elapsed_total = time.monotonic() - t_total
        print(f"\n{'─' * 50}")
        print(f"✅ All done in {format_timestamp(elapsed_total)}")
        summary = f"   📊 {processed} extracted, {skipped} skipped"
        if failed:
            summary += f", {failed} failed"
        summary += f", {total} total"
        print(summary)
    else:
        print(f"Error: {target} is not a file or directory", file=sys.stderr)
        sys.exit(1)


def _format_hms(total_secs: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(total_secs) // 3600
    m = (int(total_secs) % 3600) // 60
    s = int(total_secs) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _run_dry_estimate(args: argparse.Namespace) -> None:
    """Print cost estimate for cloud transcription without calling the API."""
    from transcriber.cloud import estimate_cost, get_duration

    target = args.input

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(
            f for f in target.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        print(f"Error: {target} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"No supported media files found in: {target}", file=sys.stderr)
        sys.exit(1)

    # Filter already-transcribed
    pending = []
    skipped = 0
    for f in files:
        if f.with_suffix(".md").exists():
            skipped += 1
        else:
            pending.append(f)

    total = len(files)
    base = target if target.is_dir() else target.parent

    print(f"\n\u2601\ufe0f  {target} (dry run)")
    print(f"   {total} media file(s), {len(pending)} to process, {skipped} already done\n")

    if not pending:
        print("   Nothing to do — all files already transcribed.")
        return

    # Gather durations
    col_width = max(len(f.relative_to(base).as_posix()) for f in pending)
    col_width = max(col_width, 4)  # minimum "File" header width

    print(f"   {'File':<{col_width}}  Duration")
    total_duration = 0.0
    for f in pending:
        dur = get_duration(f)
        total_duration += dur
        name = f.relative_to(base).as_posix() if target.is_dir() else f.name
        print(f"   {name:<{col_width}}  {_format_hms(dur)}")

    total_tokens, total_cost = estimate_cost(total_duration, args.cloud_model)
    total_minutes = total_duration / 60

    print(f"   {'─' * (col_width + 12)}")
    print(f"   {'Total duration:':<{col_width + 2}} {_format_hms(total_duration)} ({total_minutes:.1f} min)")
    print(f"   {'Estimated tokens:':<{col_width + 2}} ~{total_tokens:,}")
    print(f"   {'Model:':<{col_width + 2}} {args.cloud_model}")
    print(f"   {'Estimated cost:':<{col_width + 2}} ~${total_cost:.2f}")


def _cloud_transcribe_file(
    input_path: Path, output_path: Path, api_key: str, model: str,
    language: str | None, timestamps: bool,
) -> float:
    """Transcribe a single file via cloud API. Returns elapsed time."""
    from transcriber.cloud import transcribe_cloud

    t = time.monotonic()
    result = transcribe_cloud(
        input_path=input_path.resolve(),
        language=language,
        api_key=api_key,
        model=model,
    )
    md = build_markdown(
        input_name=input_path.name,
        paragraphs=result.paragraphs,
        timestamps=timestamps,
        language=result.language,
        lang_prob=1.0,
        duration=result.duration,
        model_name=result.model,
    )
    output_path.write_text(md, encoding="utf-8")
    print(f"  💾 Saved: {output_path.name}")
    return time.monotonic() - t


def _run_cloud_transcription(args: argparse.Namespace) -> None:
    """Handle --cloud mode: transcribe via OpenRouter API."""
    if args.dry:
        _run_dry_estimate(args)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        print("Get your key at https://openrouter.ai/keys", file=sys.stderr)
        sys.exit(1)

    if args.input.is_file():
        print(f"\n☁️  {args.input.name}")
        output_path = args.output or args.input.with_suffix(".md")
        elapsed = _cloud_transcribe_file(
            args.input, output_path, api_key, args.cloud_model,
            args.language, args.timestamps,
        )
        print(f"\n{'─' * 50}")
        print(f"✅ All done in {format_timestamp(elapsed)}")

    elif args.input.is_dir():
        files = sorted(
            f for f in args.input.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"No supported media files found in: {args.input}", file=sys.stderr)
            sys.exit(1)

        # Filter already-transcribed files
        pending = []
        skipped = 0
        for f in files:
            if f.with_suffix(".md").exists():
                skipped += 1
            else:
                pending.append(f)

        total = len(files)
        print(f"\n☁️  {args.input}")
        print(f"   {total} media file(s) found, {len(pending)} to process, {skipped} already done\n")
        print(f"{'─' * 50}")

        if not pending:
            print("✅ Nothing to do — all files already transcribed.")
            return

        processed = 0
        failed = 0
        t_total = time.monotonic()

        def _do_one(f: Path) -> tuple[Path, bool, str]:
            output_path = f.with_suffix(".md")
            try:
                _cloud_transcribe_file(
                    f, output_path, api_key, args.cloud_model,
                    args.language, args.timestamps,
                )
                return f, True, ""
            except Exception as e:
                return f, False, str(e)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_do_one, f): f for f in pending}
            for future in as_completed(futures):
                f, ok, err = future.result()
                if ok:
                    processed += 1
                else:
                    failed += 1
                    print(f"  ❌ Failed: {f.name} — {err}", file=sys.stderr)

        elapsed_total = time.monotonic() - t_total
        print(f"\n{'─' * 50}")
        print(f"✅ All done in {format_timestamp(elapsed_total)}")
        summary = f"   📊 {processed} transcribed, {skipped} skipped"
        if failed:
            summary += f", {failed} failed"
        summary += f", {total} total"
        print(summary)
    else:
        print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
        sys.exit(1)


def _run_transcription(args: argparse.Namespace) -> None:
    """Handle default mode: transcribe file(s) to markdown."""
    model = load_model(args.model)

    if args.input.is_file():
        print(f"\n📄 {args.input.name}")
        output_path = args.output or args.input.with_suffix(".md")
        elapsed = transcribe_file(args.input, output_path, model, args.language, args.timestamps, args.batch_size)
        print(f"\n{'─' * 50}")
        print(f"✅ All done in {format_timestamp(elapsed)}")
    elif args.input.is_dir():
        files = sorted(
            f for f in args.input.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"No supported media files found in: {args.input}", file=sys.stderr)
            sys.exit(1)

        total = len(files)
        skipped = 0
        processed = 0
        t_total = time.monotonic()

        print(f"\n📂 {args.input}")
        print(f"   {total} media file(s) found\n")
        print(f"{'─' * 50}")

        for i, f in enumerate(files, 1):
            output_path = f.with_suffix(".md")
            if output_path.exists():
                skipped += 1
                print(f"\n⏭️  [{i}/{total}] {f.name} — skipped (already transcribed)")
                continue
            processed += 1
            print(f"\n🎬 [{i}/{total}] {f.name}")
            transcribe_file(f, output_path, model, args.language, args.timestamps, args.batch_size)

        elapsed_total = time.monotonic() - t_total
        print(f"\n{'─' * 50}")
        print(f"✅ All done in {format_timestamp(elapsed_total)}")
        print(f"   📊 {processed} transcribed, {skipped} skipped, {total} total")
    else:
        print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
