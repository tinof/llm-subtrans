import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from filelock import FileLock
from rich.console import Console
from rich.logging import RichHandler

from PySubtrans.MKV import (
    MKVConfig,
    TranslationMode,
    VideoFile,
    get_mkv_subtitle_tracks,
    select_track_interactively,
    select_best_track_with_fallback,
    extract_track_with_progress,
    preprocess_timecodes,
    postprocess_timecodes,
    filter_subtitles,
    run_diagnostics
)

from PySubtrans import init_translator
from PySubtrans.Options import Options
from PySubtrans.SubtitleProject import SubtitleProject

# Configure rich console and logging
console = Console()
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("exsubs")


def verify_dependencies():
    """Verify required command line tools are available"""
    import subprocess
    required_commands = ["mkvextract", "mkvmerge"]
    for cmd in required_commands:
        result = subprocess.run(["which", cmd], capture_output=True)
        if result.returncode != 0:
            logger.error(f"{cmd} not found. Please install mkvtoolnix package.")
            sys.exit(1)


def verify_api_key(mode : TranslationMode):
    """Verify required API key is set based on translation mode"""
    from PySubtrans.MKV.Config import MODE_TO_ENV
    required_key = MODE_TO_ENV[mode]
    if not os.getenv(required_key):
        logger.error(f"{required_key} not found in environment")
        sys.exit(1)


def process_video_file(video_file : Path, config : MKVConfig, mode : TranslationMode, interactive : bool, show_progress : bool):
    """Process a single video file - extract and translate subtitles"""
    logger.info(f"Processing {video_file}")

    # Only process MKV files
    if not video_file.suffix.lower() == ".mkv":
        logger.warning(f"Skipping {video_file} - only MKV files are supported")
        return

    subtitle_file = video_file.with_suffix(".srt")
    lang_code = config.get_language_code(config.target_language)
    translated_file = video_file.with_suffix(f".{lang_code}.srt")

    # Check if translated file already exists
    if translated_file.exists():
        logger.info(f"Skipping {video_file} - translated subtitle already exists")
        return

    # Extract subtitles if needed
    if not subtitle_file.exists():
        if not extract_subtitles(video_file, subtitle_file, interactive, show_progress):
            return

    # Process subtitles: preprocess → filter → postprocess
    preprocess_timecodes(subtitle_file)
    filter_subtitles(subtitle_file)
    postprocess_timecodes(subtitle_file)

    # Translate subtitles using PySubtrans API
    try:
        translate_subtitles(subtitle_file, translated_file, config, mode)

        # Clean up the original extracted subtitle if translation succeeded
        if translated_file.exists() and subtitle_file.exists():
            subtitle_file.unlink()

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise


def extract_subtitles(video_file : Path, subtitle_file : Path, interactive : bool, show_progress : bool) -> bool:
    """Extract subtitles from MKV file using mkvextract"""
    if not video_file.suffix.lower() == ".mkv":
        logger.error(f"Only MKV files are supported, got: {video_file.suffix}")
        return False

    # Get subtitle tracks from MKV file
    subtitle_tracks = get_mkv_subtitle_tracks(video_file)

    if not subtitle_tracks:
        logger.error(f"No subtitle tracks found in {video_file}")
        return False

    # Select track based on mode (interactive or automatic)
    if interactive:
        selected_track = select_track_interactively(subtitle_tracks, console)
    else:
        selected_track = select_best_track_with_fallback(subtitle_tracks)

    if not selected_track:
        if interactive:
            logger.info("No subtitle track selected")
        else:
            logger.error(f"No suitable subtitle track found in {video_file} (tried English and French)")
            logger.info("Available tracks:")
            for track in subtitle_tracks:
                title = track.get("tags", {}).get("title", "No title")
                language = track.get("tags", {}).get("language", "Unknown")
                logger.info(f"  Track {track['id']}: {title} ({language})")
        return False

    # Extract using mkvextract with progress bar
    return extract_track_with_progress(
        video_file, subtitle_file, selected_track["id"], show_progress, console
    )


def translate_subtitles(sub_file : Path, out_file : Path, config : MKVConfig, mode : TranslationMode):
    """Translate subtitles using PySubtrans API"""
    from PySubtrans.MKV.Config import MODE_TO_DEFAULT_MODEL, MODE_TO_RATE_LIMIT

    # Map mode to provider name
    provider_map = {
        TranslationMode.GEMINI: "Gemini",
        TranslationMode.CHATGPT: "ChatGPT",
        TranslationMode.CLAUDE: "Claude",
        TranslationMode.DEEPSEEK: "DeepSeek",
    }

    provider = provider_map.get(mode, "Gemini")
    model = MODE_TO_DEFAULT_MODEL[mode]

    # Create PySubtrans options
    settings = {
        'provider': provider,
        'target_language': config.target_language,
        'temperature': 0.2,
        'preprocess_subtitles': True,
        'postprocess_subtitles': True,
        'model': model,
        'scene_threshold': 60.0,
        'min_batch_size': 10,
        'max_batch_size': 50,
    }

    # Set instruction file if it exists
    if config.instruction_file and config.instruction_file.exists():
        settings['instructionfile'] = str(config.instruction_file)

    options = Options(settings)

    # Create subtitle project
    project = SubtitleProject(persistent=False)
    project.InitialiseProject(str(sub_file), str(out_file))
    project.UpdateProjectSettings(options)

    # Batch subtitles for translation
    from PySubtrans import batch_subtitles
    batch_subtitles(
        project.subtitles,
        scene_threshold=60.0,
        min_batch_size=10,
        max_batch_size=50
    )

    # Translate
    translator = init_translator(options)
    project.TranslateSubtitles(translator)


def process_directory(config : MKVConfig, mode : TranslationMode, interactive : bool, show_progress : bool):
    """Process all MKV files in the current directory"""
    # Get all MKV files and sort them
    video_files = list(Path().glob("*.mkv"))

    # Filter out files that don't actually exist (symbolic links, etc.)
    existing_files = [f for f in video_files if f.exists() and f.is_file()]

    # Convert to VideoFile objects and sort
    sorted_files = sorted([VideoFile(f) for f in existing_files])

    if not sorted_files:
        logger.warning("No MKV files found in current directory")
        return

    console.print("\n[bold]Starting subtitle processing[/bold]")

    # Track statistics
    stats = {"processed": 0, "skipped": 0, "failed": 0}

    for video_file in sorted_files:
        try:
            lang_code = config.get_language_code(config.target_language)
            translated_file = video_file.path.with_suffix(f".{lang_code}.srt")

            if translated_file.exists():
                console.print(
                    f"[yellow]⏭ Skipping[/yellow] {video_file.path.name} - translation exists"
                )
                stats["skipped"] += 1
            else:
                console.print(
                    f"[blue]⚙ Processing[/blue] {video_file.path.name}"
                )
                process_video_file(video_file.path, config, mode, interactive, show_progress)
                if translated_file.exists():
                    stats["processed"] += 1
                    console.print(
                        f"[green]✓ Completed[/green] {video_file.path.name}"
                    )
                else:
                    stats["failed"] += 1
                    console.print(
                        f"[red]✗ Failed[/red] {video_file.path.name}"
                    )

        except Exception as e:
            stats["failed"] += 1
            logger.error(
                f"Error processing {video_file.path.name}: {str(e)}"
            )

        console.print("─" * 40)

    # Print summary
    console.print("\n[bold]Processing Summary[/bold]")
    console.print(f"Processed: {stats['processed']}")
    console.print(f"Skipped: {stats['skipped']}")
    console.print(f"Failed: {stats['failed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and translate subtitles from MKV files"
    )
    parser.add_argument("file", nargs="?", help="MKV file to process (optional)")

    # Create a mutually exclusive group for translation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gemini",
        action="store_const",
        dest="mode",
        const=TranslationMode.GEMINI,
        help="Use Gemini model",
    )
    mode_group.add_argument(
        "--gpt",
        action="store_const",
        dest="mode",
        const=TranslationMode.CHATGPT,
        help="Use ChatGPT model",
    )
    mode_group.add_argument(
        "--claude",
        action="store_const",
        dest="mode",
        const=TranslationMode.CLAUDE,
        help="Use Claude model",
    )
    mode_group.add_argument(
        "--deepseek",
        action="store_const",
        dest="mode",
        const=TranslationMode.DEEPSEEK,
        help="Use DeepSeek model",
    )

    parser.add_argument("-l", "--language", help="Target language")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode - manually select subtitle track to extract"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run system diagnostics to identify performance bottlenecks"
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Run diagnostics if requested
    if args.diagnose:
        run_diagnostics(console)
        return 0

    # Initialize config
    config = MKVConfig()
    if args.language:
        if args.language not in MKVConfig.SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported language: {args.language}")
            logger.info(f"Supported languages: {', '.join(MKVConfig.SUPPORTED_LANGUAGES)}")
            sys.exit(1)
        config.target_language = args.language

    # Determine translation mode
    mode = args.mode or config.default_translation_mode

    # Verify dependencies
    verify_dependencies()
    verify_api_key(mode)

    # Use file lock with proper error handling
    lock_dir = os.path.expanduser("~/.cache")
    os.makedirs(lock_dir, exist_ok=True)
    lock = FileLock(os.path.join(lock_dir, "exsubs.lock"))

    try:
        with lock.acquire(timeout=5):
            try:
                # Create processor with progress control
                show_progress = not args.no_progress

                if args.file:
                    # Process single file
                    process_video_file(Path(args.file), config, mode, args.interactive, show_progress)
                else:
                    # Process all MKV files in current directory
                    process_directory(config, mode, args.interactive, show_progress)

                return 0

            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                return 1
    except TimeoutError:
        logging.error("Could not acquire lock - another instance might be running")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return 1
    finally:
        if lock.is_locked:
            lock.release()


if __name__ == '__main__':
    raise SystemExit(main())
