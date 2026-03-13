import argparse
import sys
import time
from pathlib import Path

from transcriber.core import SUPPORTED_EXTENSIONS, format_timestamp, load_model, transcribe


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
               "  transcribe podcast.mp3 -o notes.md --timestamps",
    )
    parser.add_argument("input", type=Path, help="Path to video/audio file or folder")
    parser.add_argument("-o", "--output", type=Path, help="Output markdown file (ignored for folders)")
    parser.add_argument("-m", "--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("-l", "--language", default=None, help="Language code, e.g. 'en', 'uk', 'ru' (default: auto-detect)")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in output (off by default)")
    parser.add_argument("-b", "--batch-size", type=int, default=4,
                        help="Batch size for inference — lower = less RAM (default: 4)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: path not found: {args.input}", file=sys.stderr)
        sys.exit(1)

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
