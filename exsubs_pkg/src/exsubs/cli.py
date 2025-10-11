import argparse
import logging
import os
import sys
from pathlib import Path

from filelock import FileLock
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from . import (
    Config,
    TranslationMode,
    VideoFile,
    SubtitleProcessor,
    run_diagnostics
)

# Configure rich console and logging
console = Console()
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("exsubs")


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
        run_diagnostics()
        return

    # Initialize config
    config = Config()
    if args.language:
        if args.language not in Config.SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported language: {args.language}")
            logger.info(f"Supported languages: {', '.join(Config.SUPPORTED_LANGUAGES)}")
            sys.exit(1)
        config.target_language = args.language

    # Determine translation mode
    mode = args.mode or config.default_translation_mode

    # Disable debug logging for production
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Use file lock with proper error handling
    lock_dir = os.path.expanduser("~/.cache")
    os.makedirs(lock_dir, exist_ok=True)
    lock = FileLock(os.path.join(lock_dir, "exsubs.lock"))

    try:
        with lock.acquire(timeout=5):
            try:
                # Create processor with progress control
                show_progress = not args.no_progress
                processor = SubtitleProcessor(config, mode, args.interactive, show_progress)

                if args.file:
                    # Process single file
                    processor.process_video(Path(args.file))
                else:
                    # Process all MKV files in current directory
                    process_directory(processor, config)

            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                sys.exit(1)
    except TimeoutError:
        logging.error("Could not acquire lock - another instance might be running")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        if lock.is_locked:
            lock.release()


def process_directory(processor: SubtitleProcessor, config: Config):
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
            translated_file = video_file.path.with_suffix(
                f".{config.get_language_code(config.target_language)}.srt"
            )
            
            if translated_file.exists():
                console.print(
                    f"[yellow]⏭ Skipping[/yellow] {video_file.path.name} - translation exists"
                )
                stats["skipped"] += 1
            else:
                console.print(
                    f"[blue]⚙ Processing[/blue] {video_file.path.name}"
                )
                processor.process_video(video_file.path)
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
