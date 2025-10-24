import argparse
import logging
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import subprocess
import shutil

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from PySubtrans.MKV import (
    preprocess_timecodes,
    postprocess_timecodes,
    filter_subtitles,
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
logger = logging.getLogger("transubs")

GEMINI_DEFAULT_MODEL = "gemini-2.5-pro"
GEMINI_SCENE_THRESHOLD = 240.0
GEMINI_MIN_BATCH_SIZE = 80
GEMINI_MAX_BATCH_SIZE = 180
GEMINI_MAX_CONTEXT_SUMMARIES = 6
GEMINI_STANDARD_RATE_LIMIT = 150.0
GEMINI_FLASH_RATE_LIMIT = 1000.0
GEMINI_FLASH_MODEL_SUFFIX = "gemini-2.5-flash-preview-09-2025"
FLASH_SCENE_THRESHOLD = 300.0
FLASH_MIN_BATCH_SIZE = 100
FLASH_MAX_BATCH_SIZE = 220


class TranslationMetrics:
    """Collect batched translation statistics for user feedback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._attached = False
        self._batch_count = 0
        self._total_lines = 0
        self._line_counts : list[int] = []
        self._prompt_tokens : list[int] = []
        self._output_tokens : list[int] = []
        self._total_tokens : list[int] = []

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

    def render(self, elapsed_seconds : float, expected_lines : int, expected_batches : int, rate_limit : float|None):
        if self._batch_count == 0:
            return

        avg_lines = statistics.fmean(self._line_counts) if self._line_counts else 0.0
        max_lines = max(self._line_counts) if self._line_counts else 0
        avg_prompt = statistics.fmean(self._prompt_tokens) if self._prompt_tokens else 0.0
        avg_output = statistics.fmean(self._output_tokens) if self._output_tokens else 0.0
        avg_total = statistics.fmean(self._total_tokens) if self._total_tokens else 0.0
        max_total = max(self._total_tokens) if self._total_tokens else 0

        lines_per_min = 0.0
        if elapsed_seconds > 0:
            lines_per_min = (self._total_lines / elapsed_seconds) * 60.0

        console.print("\n[bold]Gemini Translation Metrics[/bold]")
        console.print(f"Batches translated: {self._batch_count}/{expected_batches or self._batch_count}")
        console.print(f"Lines processed: {self._total_lines}/{expected_lines or self._total_lines}")
        console.print(f"Lines per batch (avg/max): {avg_lines:.1f}/{max_lines}")
        if self._total_tokens:
            console.print(f"Token usage avg (prompt/output/total): {avg_prompt:.0f}/{avg_output:.0f}/{avg_total:.0f}")
            console.print(f"Peak total tokens: {max_total}")
        console.print(f"Throughput: {lines_per_min:.1f} lines/min ({elapsed_seconds:.1f}s elapsed)")
        if rate_limit:
            console.print(f"Applied rate limit: {rate_limit:.0f} RPM")

    def _on_batch_translated(self, _sender, batch) -> None:
        with self._lock:
            self._batch_count += 1
            self._total_lines += batch.size or 0
            self._line_counts.append(batch.size or 0)
            tr = getattr(batch, 'translation', None)
            if tr and isinstance(tr.content, dict):
                for key, bucket in (("prompt_tokens", self._prompt_tokens), ("output_tokens", self._output_tokens), ("total_tokens", self._total_tokens)):
                    val = tr.content.get(key)
                    if isinstance(val, (int, float)):
                        bucket.append(int(val))


def _normalise_model_name(model : str|None) -> str|None:
    if not model:
        return None
    return model.lower().split('/')[-1]


def _determine_gemini_rate_limit(model : str|None) -> float:
    normalised = _normalise_model_name(model)
    if normalised == GEMINI_FLASH_MODEL_SUFFIX:
        return GEMINI_FLASH_RATE_LIMIT
    return GEMINI_STANDARD_RATE_LIMIT


def _using_vertex() -> bool:
    value = os.getenv("GEMINI_USE_VERTEX")
    if value is None:
        return True
    return value.lower() in {"1", "true", "yes", "on"}


def _detect_gcloud_project() -> str|None:
    for key in ("VERTEX_PROJECT", "GEMINI_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        value = os.getenv(key)
        if value:
            return value
    try:
        result = subprocess.run(["gcloud", "config", "get-value", "project", "--quiet"], capture_output=True, text=True, timeout=2)
        proj = (result.stdout or "").strip()
        if proj and proj != "(unset)":
            return proj
    except Exception:
        pass
    return None


def _vertex_project() -> str|None:
    return _detect_gcloud_project()


def _vertex_location() -> str|None:
    return os.getenv("VERTEX_LOCATION") or os.getenv("GEMINI_VERTEX_LOCATION") or "europe-west1"


def translate_srt_file(sub_file : Path, target_language : str|None, provider_flag : str|None) -> None:
    from PySubtrans.MKV.Config import MODE_TO_DEFAULT_MODEL, MODE_TO_RATE_LIMIT, TranslationMode, MKVConfig

    # Preprocess the input file in place
    preprocess_timecodes(sub_file)
    filter_subtitles(sub_file)
    postprocess_timecodes(sub_file)

    # Provider mapping
    provider_map = {
        TranslationMode.GEMINI: "Gemini",
        TranslationMode.CHATGPT: "ChatGPT",
        TranslationMode.CLAUDE: "Claude",
        TranslationMode.DEEPSEEK: "DeepSeek",
    }

    mode = {
        "gemini": TranslationMode.GEMINI,
        "gpt": TranslationMode.CHATGPT,
        "claude": TranslationMode.CLAUDE,
        "deepseek": TranslationMode.DEEPSEEK,
    }.get(provider_flag or "gemini", TranslationMode.GEMINI)

    provider = provider_map.get(mode, "Gemini")
    model = MODE_TO_DEFAULT_MODEL[mode]

    # Tuned defaults and rate limits
    scene_threshold = float(os.getenv('SCENE_THRESHOLD') or 60.0)
    min_batch_size = int(os.getenv('MIN_BATCH_SIZE') or 10)
    max_batch_size = int(os.getenv('MAX_BATCH_SIZE') or 50)
    max_context_summaries = int(os.getenv('MAX_CONTEXT_SUMMARIES') or 10)
    rate_limit : float|None = None

    if mode == TranslationMode.GEMINI:
        model = os.getenv('GEMINI_MODEL') or GEMINI_DEFAULT_MODEL
        normalised = _normalise_model_name(model)
        if normalised == GEMINI_FLASH_MODEL_SUFFIX:
            scene_threshold = float(os.getenv('SCENE_THRESHOLD') or FLASH_SCENE_THRESHOLD)
            min_batch_size = int(os.getenv('MIN_BATCH_SIZE') or FLASH_MIN_BATCH_SIZE)
            max_batch_size = int(os.getenv('MAX_BATCH_SIZE') or FLASH_MAX_BATCH_SIZE)
        else:
            scene_threshold = float(os.getenv('SCENE_THRESHOLD') or GEMINI_SCENE_THRESHOLD)
            min_batch_size = int(os.getenv('MIN_BATCH_SIZE') or GEMINI_MIN_BATCH_SIZE)
            max_batch_size = int(os.getenv('MAX_BATCH_SIZE') or GEMINI_MAX_BATCH_SIZE)
        max_context_summaries = int(os.getenv('MAX_CONTEXT_SUMMARIES') or GEMINI_MAX_CONTEXT_SUMMARIES)
        rate_limit = _determine_gemini_rate_limit(model)
        os.environ.setdefault('GEMINI_RATE_LIMIT', str(int(rate_limit)))
    else:
        rate_value = MODE_TO_RATE_LIMIT.get(mode)
        if rate_value:
            try:
                rate_limit = float(rate_value)
            except ValueError:
                rate_limit = None

    # Vertex settings
    settings_vertex : dict[str,str|bool] = {}
    if mode == TranslationMode.GEMINI:
        use_vertex = _using_vertex()
        settings_vertex['use_vertex'] = use_vertex
        if use_vertex:
            project = _vertex_project()
            location = _vertex_location()
            if project:
                settings_vertex['vertex_project'] = project
            if location:
                settings_vertex['vertex_location'] = location

    # Build options
    settings = {
        'provider': provider,
        'target_language': target_language or MKVConfig().target_language,
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
    settings.update(settings_vertex)

    options = Options(settings)

    # Create project on the input file
    project = SubtitleProject(persistent=False)
    output = None
    if target_language:
        # Decide output path like exsubs: add language code suffix
        lang_code = MKVConfig.get_language_code(target_language)
        output = str(sub_file.with_suffix(f".{lang_code}.srt"))
    project.InitialiseProject(str(sub_file), output)
    project.UpdateProjectSettings(options)

    # Batch subtitles
    from PySubtrans import batch_subtitles
    batch_subtitles(
        project.subtitles,
        scene_threshold=scene_threshold,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size
    )

    total_batches = sum(len(scene.batches) for scene in project.subtitles.scenes) if project.subtitles and project.subtitles.scenes else 0
    total_lines = project.subtitles.linecount if project.subtitles else 0

    translator = init_translator(options)
    metrics = TranslationMetrics() if mode == TranslationMode.GEMINI else None
    start = time.perf_counter()
    try:
        if metrics:
            metrics.attach(translator)
        project.TranslateSubtitles(translator)
    finally:
        if metrics:
            metrics.detach(translator)
            metrics.render(time.perf_counter() - start, total_lines, total_batches, rate_limit)


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate an .srt (or supported) subtitle file using tuned exsubs defaults")
    parser.add_argument("file", help="Subtitle file to translate (.srt, .ass, .vtt)")
    # provider choice similar to exsubs
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gemini", action="store_true", help="Use Gemini model (default)")
    mode_group.add_argument("--gpt", action="store_true", help="Use ChatGPT model")
    mode_group.add_argument("--claude", action="store_true", help="Use Claude model")
    mode_group.add_argument("--deepseek", action="store_true", help="Use DeepSeek model")
    parser.add_argument("-l", "--language", help="Target language (default from config)")
    parser.add_argument("--setup-vertex", action="store_true", help="Run Vertex setup assistant (delegates to exsubs --setup-vertex)")
    parser.add_argument("-y", "--yes", action="store_true", help="Answer yes to prompts for setup assistant")

    args = parser.parse_args()

    load_dotenv()

    if args.setup_vertex:
        # Reuse the setup assistant from exsubs
        try:
            from scripts.exsubs import setup_vertex
            return setup_vertex(yes=args.yes)
        except Exception as e:
            console.print(f"[red]Failed to run setup assistant:[/red] {e}")
            return 1

    sub_path = Path(args.file)
    if not sub_path.exists() or not sub_path.is_file():
        logger.error(f"Subtitle file not found: {sub_path}")
        return 1

    provider_flag = "gemini"
    if args.gpt:
        provider_flag = "gpt"
    elif args.claude:
        provider_flag = "claude"
    elif args.deepseek:
        provider_flag = "deepseek"

    try:
        translate_srt_file(sub_path, args.language, provider_flag)
        return 0
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

