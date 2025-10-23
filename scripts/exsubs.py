import argparse
import logging
import os
from pathlib import Path
import statistics
import sys
import threading
import time

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

GEMINI_DEFAULT_MODEL = "gemini-2.5-pro"
GEMINI_SCENE_THRESHOLD = 240.0
GEMINI_MIN_BATCH_SIZE = 80
GEMINI_MAX_BATCH_SIZE = 180
GEMINI_STANDARD_RATE_LIMIT = 150.0
GEMINI_FLASH_RATE_LIMIT = 1000.0
GEMINI_FLASH_MODEL_SUFFIX = "gemini-2.5-flash-preview-09-2025"
GEMINI_MAX_CONTEXT_SUMMARIES = 6


class TranslationMetrics:
    """Collect batched translation statistics for user feedback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._attached = False
        self._batch_count = 0
        self._total_lines = 0
        self._line_counts: list[int] = []
        self._prompt_tokens: list[int] = []
        self._output_tokens: list[int] = []
        self._total_tokens: list[int] = []

    def attach(self, translator):
        if self._attached:
            return
        translator.events.batch_translated.connect(self._on_batch_translated)
        self._attached = True

    def detach(self, translator):
        if not self._attached:
            return
        translator.events.batch_translated.disconnect(self._on_batch_translated)
        self._attached = False

    def has_data(self) -> bool:
        return self._batch_count > 0

    def render(self, elapsed_seconds: float, expected_lines: int, expected_batches: int, rate_limit: float|None):
        if not self.has_data():
            return

        avg_lines = statistics.fmean(self._line_counts) if self._line_counts else 0.0
        max_lines = max(self._line_counts) if self._line_counts else 0
        avg_prompt = statistics.fmean(self._prompt_tokens) if self._prompt_tokens else 0.0
        avg_output = statistics.fmean(self._output_tokens) if self._output_tokens else 0.0
        avg_total = statistics.fmean(self._total_tokens) if self._total_tokens else 0.0
        max_total = max(self._total_tokens) if self._total_tokens else 0

        lines_per_minute = 0.0
        if elapsed_seconds > 0:
            lines_per_minute = (self._total_lines / elapsed_seconds) * 60.0

        console.print("\n[bold]Gemini Translation Metrics[/bold]")
        console.print(f"Batches translated: {self._batch_count}/{expected_batches or self._batch_count}")
        console.print(f"Lines processed: {self._total_lines}/{expected_lines or self._total_lines}")
        console.print(f"Lines per batch (avg/max): {avg_lines:.1f}/{max_lines}")
        if self._total_tokens:
            console.print(f"Token usage avg (prompt/output/total): {avg_prompt:.0f}/{avg_output:.0f}/{avg_total:.0f}")
            console.print(f"Peak total tokens: {max_total}")
        console.print(f"Throughput: {lines_per_minute:.1f} lines/min ({elapsed_seconds:.1f}s elapsed)")
        if rate_limit:
            console.print(f"Applied rate limit: {rate_limit:.0f} RPM")

    def _on_batch_translated(self, _sender, batch) -> None:
        with self._lock:
            line_count = batch.size or 0
            self._batch_count += 1
            self._total_lines += line_count
            self._line_counts.append(line_count)

            translation = getattr(batch, "translation", None)
            if translation and isinstance(translation.content, dict):
                self._append_token(self._prompt_tokens, translation.content.get('prompt_tokens'))
                self._append_token(self._output_tokens, translation.content.get('output_tokens'))
                self._append_token(self._total_tokens, translation.content.get('total_tokens'))

    def _append_token(self, bucket: list[int], value):
        if isinstance(value, (int, float)):
            bucket.append(int(value))


def _normalise_model_name(model: str|None) -> str|None:
    if not model:
        return None
    return model.lower().split('/')[-1]


def _determine_gemini_rate_limit(model: str|None) -> float:
    normalised = _normalise_model_name(model)
    if normalised == GEMINI_FLASH_MODEL_SUFFIX:
        return GEMINI_FLASH_RATE_LIMIT
    return GEMINI_STANDARD_RATE_LIMIT


def _using_vertex() -> bool:
    value = os.getenv("GEMINI_USE_VERTEX")
    if value is None:
        return True

    return value.lower() in {"1", "true", "yes", "on"}


def _vertex_project() -> str|None:
    return os.getenv("VERTEX_PROJECT") or os.getenv("GEMINI_VERTEX_PROJECT")


def _vertex_location() -> str|None:
    return os.getenv("VERTEX_LOCATION") or os.getenv("GEMINI_VERTEX_LOCATION") or "us-central1"


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
    if mode == TranslationMode.GEMINI and _using_vertex():
        return

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
    # Defaults; overridable by environment or model-specific logic below
    scene_threshold = float(os.getenv('SCENE_THRESHOLD') or 60.0)
    min_batch_size = int(os.getenv('MIN_BATCH_SIZE') or 10)
    max_batch_size = int(os.getenv('MAX_BATCH_SIZE') or 50)
    max_context_summaries = int(os.getenv('MAX_CONTEXT_SUMMARIES') or 10)
    rate_limit : float|None = None

    if mode == TranslationMode.GEMINI:
        model = os.getenv('GEMINI_MODEL') or GEMINI_DEFAULT_MODEL
        scene_threshold = float(os.getenv('SCENE_THRESHOLD') or GEMINI_SCENE_THRESHOLD)
        min_batch_size = int(os.getenv('MIN_BATCH_SIZE') or GEMINI_MIN_BATCH_SIZE)
        max_batch_size = int(os.getenv('MAX_BATCH_SIZE') or GEMINI_MAX_BATCH_SIZE)
        max_context_summaries = int(os.getenv('MAX_CONTEXT_SUMMARIES') or GEMINI_MAX_CONTEXT_SUMMARIES)
        rate_limit = _determine_gemini_rate_limit(model)

        os.environ.setdefault('GEMINI_RATE_LIMIT', str(int(rate_limit)))

        use_vertex = _using_vertex()
        settings_vertex : dict[str,str|bool] = {'use_vertex': use_vertex}
        if use_vertex:
            project = _vertex_project()
            location = _vertex_location()
            if project:
                settings_vertex['vertex_project'] = project
            if location:
                settings_vertex['vertex_location'] = location
    else:
        rate_value = MODE_TO_RATE_LIMIT.get(mode)
        if rate_value:
            try:
                rate_limit = float(rate_value)
            except ValueError:
                rate_limit = None
        settings_vertex = {}

    # Create PySubtrans options
    settings = {
        'provider': provider,
        'target_language': config.target_language,
        'temperature': 0.2,
        'preprocess_subtitles': True,
        'postprocess_subtitles': True,
        'model': model,
        'scene_threshold': scene_threshold,
        'min_batch_size': min_batch_size,
        'max_batch_size': max_batch_size,
        'max_context_summaries': max_context_summaries,
    }

    if rate_limit:
        settings['rate_limit'] = rate_limit

    if settings_vertex:
        settings.update(settings_vertex)

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
        scene_threshold=scene_threshold,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size
    )

    total_batches = sum(len(scene.batches) for scene in project.subtitles.scenes) if project.subtitles and project.subtitles.scenes else 0
    total_lines = project.subtitles.linecount if project.subtitles else 0

    # Translate
    translator = init_translator(options)
    metrics = TranslationMetrics() if mode == TranslationMode.GEMINI else None
    start_time = time.perf_counter()
    try:
        if metrics:
            metrics.attach(translator)
        project.TranslateSubtitles(translator)
    finally:
        if metrics:
            metrics.detach(translator)
            elapsed = time.perf_counter() - start_time
            metrics.render(elapsed, total_lines, total_batches, rate_limit)


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
