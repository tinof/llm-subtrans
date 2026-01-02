import argparse
import logging
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import subprocess

from dotenv import load_dotenv
import regex
from rich.console import Console
from rich.logging import RichHandler

from PySubtrans.MKV import (
    preprocess_timecodes,
    postprocess_timecodes,
    filter_subtitles,
)
from PySubtrans.MKV.Config import TranslationMode

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

LANGUAGE_SUFFIX_PATTERN = regex.compile(
    r"^\.[a-z]{2,3}(?:-[a-z]{2,3})?$", regex.IGNORECASE
)


def _looks_like_language_suffix(suffix: str) -> bool:
    """Return True if suffix resembles a language identifier."""
    return bool(LANGUAGE_SUFFIX_PATTERN.match(suffix))


def _build_translated_output_path(subtitle_file: Path, lang_code: str) -> Path:
    extension = subtitle_file.suffix or ".srt"
    filename = subtitle_file.name
    lang_code = lang_code.lower()

    if extension:
        base_name = filename[: -len(extension)]
    else:
        base_name = filename

    while True:
        dot_index = base_name.rfind(".")
        if dot_index == -1:
            break
        candidate = base_name[dot_index:]
        if not _looks_like_language_suffix(candidate):
            break
        base_name = base_name[:dot_index]

    return subtitle_file.parent / f"{base_name}.{lang_code}{extension}"


def _normalise_translated_output(
    subtitle_file: Path, desired_file: Path, target_language: str | None, lang_code: str
) -> Path:
    if desired_file.exists():
        return desired_file
    suffix = subtitle_file.suffix or ".srt"
    stem_with_lang = subtitle_file.stem
    candidates: list[Path] = [
        subtitle_file.with_name(f"{stem_with_lang}.translated{suffix}"),
        subtitle_file.with_name(f"{stem_with_lang}.{lang_code}{suffix}"),
    ]
    if target_language:
        candidates.append(
            subtitle_file.with_name(
                f"{stem_with_lang}.{target_language.lower()}{suffix}"
            )
        )
    seen: set[str] = set()
    ordered: list[Path] = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            ordered.append(c)
            seen.add(s)
    for c in ordered:
        if c.exists():
            try:
                c.rename(desired_file)
                logger.info(
                    f"Renamed translated subtitles from {c.name} to {desired_file.name}"
                )
                return desired_file
            except Exception as exc:
                logger.error(
                    f"Failed to rename translated subtitles {c} -> {desired_file}: {exc}"
                )
                return c
    return desired_file


def _fix_finnish_subtitles(translated_file: Path) -> None:
    if not translated_file.exists():
        logger.warning(f"Cannot fix subtitles; {translated_file} does not exist")
        return
    try:
        subprocess.run(["fix-finnish-subs", str(translated_file)], check=True)
        logger.info(f"Fixed subtitles with fix-finnish-subs: {translated_file.name}")
    except subprocess.CalledProcessError as exc:
        logger.error(f"fix-finnish-subs failed for {translated_file}: {exc}")
    except Exception as exc:
        logger.error(
            f"Unexpected error running fix-finnish-subs for {translated_file}: {exc}"
        )


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

    def render(
        self,
        elapsed_seconds: float,
        expected_lines: int,
        expected_batches: int,
        rate_limit: float | None,
    ):
        if self._batch_count == 0:
            return

        avg_lines = statistics.fmean(self._line_counts) if self._line_counts else 0.0
        max_lines = max(self._line_counts) if self._line_counts else 0
        avg_prompt = (
            statistics.fmean(self._prompt_tokens) if self._prompt_tokens else 0.0
        )
        avg_output = (
            statistics.fmean(self._output_tokens) if self._output_tokens else 0.0
        )
        avg_total = statistics.fmean(self._total_tokens) if self._total_tokens else 0.0
        max_total = max(self._total_tokens) if self._total_tokens else 0

        lines_per_min = 0.0
        if elapsed_seconds > 0:
            lines_per_min = (self._total_lines / elapsed_seconds) * 60.0

        console.print("\n[bold]Gemini Translation Metrics[/bold]")
        console.print(
            f"Batches translated: {self._batch_count}/{expected_batches or self._batch_count}"
        )
        console.print(
            f"Lines processed: {self._total_lines}/{expected_lines or self._total_lines}"
        )
        console.print(f"Lines per batch (avg/max): {avg_lines:.1f}/{max_lines}")
        if self._total_tokens:
            console.print(
                f"Token usage avg (prompt/output/total): {avg_prompt:.0f}/{avg_output:.0f}/{avg_total:.0f}"
            )
            console.print(f"Peak total tokens: {max_total}")
        console.print(
            f"Throughput: {lines_per_min:.1f} lines/min ({elapsed_seconds:.1f}s elapsed)"
        )
        if rate_limit:
            console.print(f"Applied rate limit: {rate_limit:.0f} RPM")

    def _on_batch_translated(self, _sender, batch) -> None:
        with self._lock:
            self._batch_count += 1
            self._total_lines += batch.size or 0
            self._line_counts.append(batch.size or 0)
            tr = getattr(batch, "translation", None)
            if tr and isinstance(tr.content, dict):
                for key, bucket in (
                    ("prompt_tokens", self._prompt_tokens),
                    ("output_tokens", self._output_tokens),
                    ("total_tokens", self._total_tokens),
                ):
                    val = tr.content.get(key)
                    if isinstance(val, (int, float)):
                        bucket.append(int(val))


class TranslationProgress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self._file: Path | None = None
        self._total_scenes = 0
        self._done_scenes = 0
        self._total_batches = 0
        self._done_batches = 0
        self._total_lines = 0
        self._done_lines = 0
        self._last_len = 0

    def attach(self, translator, file_path: Path, total_lines: int):
        self._file = file_path
        self._total_lines = total_lines
        translator.events.preprocessed.connect(self._on_pre)
        translator.events.batch_translated.connect(self._on_batch)
        translator.events.scene_translated.connect(self._on_scene)

    def detach(self, translator, final: bool = False):
        try:
            translator.events.preprocessed.disconnect(self._on_pre)
            translator.events.batch_translated.disconnect(self._on_batch)
            translator.events.scene_translated.disconnect(self._on_scene)
        except Exception:
            pass
        self._render(final=True)

    def _on_pre(self, _s, scenes):
        self._total_scenes = len(scenes)
        self._total_batches = sum(len(sc.batches) for sc in scenes)
        self._render()

    def _on_batch(self, _s, batch):
        self._done_batches += 1
        self._done_lines += batch.size or 0
        self._render()

    def _on_scene(self, _s, **kwargs):
        # Accept keyword argument 'scene' from signal
        self._done_scenes += 1
        self._render()

    def _render(self, final: bool = False):
        if not self._file:
            return
        parts = [
            f"Translating {self._file.name}",
            f"scenes {self._done_scenes}/{self._total_scenes}",
            f"batches {self._done_batches}/{self._total_batches}",
        ]
        if self._total_lines:
            parts.append(f"lines {self._done_lines}/{self._total_lines}")
        msg = " | ".join(parts)
        pad = ""
        if len(msg) < self._last_len:
            pad = " " * (self._last_len - len(msg))
        self.stream.write(msg + pad + ("\n" if final else "\r"))
        self.stream.flush()
        self._last_len = len(msg)


def _normalise_model_name(model: str | None) -> str | None:
    if not model:
        return None
    return model.lower().split("/")[-1]


def _determine_gemini_rate_limit(model: str | None) -> float:
    normalised = _normalise_model_name(model)
    if normalised == GEMINI_FLASH_MODEL_SUFFIX:
        return GEMINI_FLASH_RATE_LIMIT
    return GEMINI_STANDARD_RATE_LIMIT


def _using_vertex() -> bool:
    value = os.getenv("GEMINI_USE_VERTEX")
    if value is None:
        return True
    return value.lower() in {"1", "true", "yes", "on"}


def _detect_gcloud_project() -> str | None:
    for key in (
        "VERTEX_PROJECT",
        "GEMINI_VERTEX_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
    ):
        value = os.getenv(key)
        if value:
            return value
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project", "--quiet"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        proj = (result.stdout or "").strip()
        if proj and proj != "(unset)":
            return proj
    except Exception:
        pass
    return None


def _vertex_project() -> str | None:
    return _detect_gcloud_project()


def _vertex_location() -> str | None:
    return (
        os.getenv("VERTEX_LOCATION")
        or os.getenv("GEMINI_VERTEX_LOCATION")
        or "europe-west1"
    )


def _env_default_mode() -> TranslationMode | None:
    """Resolve translation mode defaults from environment for transubs."""
    raw = os.getenv("LLMSUBTRANS_DEFAULT_MODE") or os.getenv("TRANSUBS_DEFAULT_MODE")
    if not raw:
        return None
    key = raw.strip().lower()
    if key in {"gpt", "openai", "chatgpt"}:
        return TranslationMode.CHATGPT
    if key in {"gemini", "google", "vertex"}:
        return TranslationMode.GEMINI
    if key in {"claude", "anthropic"}:
        return TranslationMode.CLAUDE
    if key in {"deepseek"}:
        return TranslationMode.DEEPSEEK
    return None


def translate_srt_file(
    sub_file: Path,
    target_language: str | None,
    provider_flag: str | None,
    show_progress: bool = True,
    show_metrics: bool = True,
    proofread: bool = False,
) -> None:
    from PySubtrans.MKV.Config import (
        MODE_TO_DEFAULT_MODEL,
        MODE_TO_RATE_LIMIT,
        MKVConfig,
    )

    # Preprocess the input file in place
    preprocess_timecodes(sub_file)
    filter_subtitles(sub_file)
    postprocess_timecodes(sub_file)

    # Provider mapping
    provider_map = {
        TranslationMode.GEMINI: "Gemini",
        TranslationMode.CHATGPT: "OpenAI",
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
    scene_threshold = float(os.getenv("SCENE_THRESHOLD") or 60.0)
    min_batch_size = int(os.getenv("MIN_BATCH_SIZE") or 10)
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE") or 50)
    max_context_summaries = int(os.getenv("MAX_CONTEXT_SUMMARIES") or 10)
    rate_limit: float | None = None

    max_context_summaries = int(os.getenv("MAX_CONTEXT_SUMMARIES") or 10)
    rate_limit: float | None = None

    # Check for large context mode
    # Default to True for Gemini, False for others, unless explicitly set
    env_large_context = os.getenv("LARGE_CONTEXT_MODE")
    if env_large_context is not None:
        large_context_mode = env_large_context.lower() in ("true", "1", "yes")
    else:
        large_context_mode = mode == TranslationMode.GEMINI

    if mode == TranslationMode.GEMINI:
        model = os.getenv("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL

        # Only apply manual tuning if large_context_mode is NOT enabled
        if not large_context_mode:
            normalised = _normalise_model_name(model)
            if normalised == GEMINI_FLASH_MODEL_SUFFIX:
                scene_threshold = float(
                    os.getenv("SCENE_THRESHOLD") or FLASH_SCENE_THRESHOLD
                )
                min_batch_size = int(
                    os.getenv("MIN_BATCH_SIZE") or FLASH_MIN_BATCH_SIZE
                )
                max_batch_size = int(
                    os.getenv("MAX_BATCH_SIZE") or FLASH_MAX_BATCH_SIZE
                )
            else:
                scene_threshold = float(
                    os.getenv("SCENE_THRESHOLD") or GEMINI_SCENE_THRESHOLD
                )
                min_batch_size = int(
                    os.getenv("MIN_BATCH_SIZE") or GEMINI_MIN_BATCH_SIZE
                )
                max_batch_size = int(
                    os.getenv("MAX_BATCH_SIZE") or GEMINI_MAX_BATCH_SIZE
                )
        else:
            # If large context mode is on, let SubtitleBatcher handle defaults (or use env overrides)
            if not os.getenv("SCENE_THRESHOLD"):
                scene_threshold = 300.0
            if not os.getenv("MAX_BATCH_SIZE"):
                max_batch_size = 600
            if not os.getenv("MIN_BATCH_SIZE"):
                min_batch_size = 1

        max_context_summaries = int(
            os.getenv("MAX_CONTEXT_SUMMARIES") or GEMINI_MAX_CONTEXT_SUMMARIES
        )
        rate_limit = _determine_gemini_rate_limit(model)
        os.environ.setdefault("GEMINI_RATE_LIMIT", str(int(rate_limit)))
    elif mode == TranslationMode.CHATGPT:
        model = os.getenv("OPENAI_MODEL") or "gpt-5-mini"
        scene_threshold = float(os.getenv("SCENE_THRESHOLD") or 120.0)
        min_batch_size = int(os.getenv("MIN_BATCH_SIZE") or 30)
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE") or 100)
        max_context_summaries = int(os.getenv("MAX_CONTEXT_SUMMARIES") or 6)
        try:
            base_rate = os.getenv("OPENAI_RATE_LIMIT") or MODE_TO_RATE_LIMIT.get(
                mode, ""
            )
            rate_limit = float(base_rate) if base_rate else None
        except ValueError:
            rate_limit = None
    else:
        rate_value = MODE_TO_RATE_LIMIT.get(mode)
        if rate_value:
            try:
                rate_limit = float(rate_value)
            except ValueError:
                rate_limit = None

    # Vertex settings
    settings_vertex: dict[str, str | bool] = {}
    if mode == TranslationMode.GEMINI:
        use_vertex = _using_vertex()
        settings_vertex["use_vertex"] = use_vertex
        if use_vertex:
            project = _vertex_project()
            location = _vertex_location()
            if project:
                settings_vertex["vertex_project"] = project
            if location:
                settings_vertex["vertex_location"] = location

    # Determine temperature
    env_temp = os.getenv("LLM_TEMPERATURE")
    if mode == TranslationMode.GEMINI:
        env_temp = os.getenv("GEMINI_TEMPERATURE") or env_temp
    elif mode == TranslationMode.CHATGPT:
        env_temp = os.getenv("OPENAI_TEMPERATURE") or env_temp
    elif mode == TranslationMode.CLAUDE:
        env_temp = os.getenv("CLAUDE_TEMPERATURE") or env_temp
    elif mode == TranslationMode.DEEPSEEK:
        env_temp = os.getenv("DEEPSEEK_TEMPERATURE") or env_temp

    try:
        temperature = float(env_temp) if env_temp else 1.0
    except ValueError:
        temperature = 1.0

    # Build options
    settings = {
        "provider": provider,
        "target_language": target_language or MKVConfig().target_language,
        "temperature": temperature,
        "preprocess_subtitles": True,
        "postprocess_subtitles": True,
        "model": model,
        "scene_threshold": scene_threshold,
        "min_batch_size": min_batch_size,
        "max_batch_size": max_batch_size,
        "max_context_summaries": max_context_summaries,
        "large_context_mode": large_context_mode,
    }
    if rate_limit:
        settings["rate_limit"] = rate_limit
    settings.update(settings_vertex)

    options = Options(settings)

    # Display configuration information
    effective_language = target_language or MKVConfig().target_language
    lang_code = MKVConfig.get_language_code(effective_language)

    console.print("\n[bold cyan]Translation Configuration:[/bold cyan]")
    console.print(f"  Provider: {provider}")
    console.print(f"  Model: {model}")
    console.print(f"  Target Language: {effective_language}")
    console.print(f"  Temperature: {temperature}")
    if rate_limit:
        console.print(f"  Rate Limit: {rate_limit:.0f} RPM")

    # Display instruction file info (if MKVConfig has one configured)
    # We need to re-resolve the instruction file if a target language is provided
    config = MKVConfig(target_language=effective_language)

    if config.instruction_file and config.instruction_file.exists():
        console.print(f"  Instructions: {config.instruction_file}")
    elif config.instruction_file:
        console.print(
            f"  Instructions: {config.instruction_file} [yellow](not found)[/yellow]"
        )
    else:
        console.print("  Instructions: [dim]None[/dim]")

    console.print()

    # Create project on the input file
    project = SubtitleProject(persistent=False)
    # Decide output path: replace existing language segment with target code
    effective_language = target_language or MKVConfig().target_language
    lang_code = MKVConfig.get_language_code(effective_language)

    if proofread:
        # Locate the proofread instructions file
        root_dir = Path(__file__).parent.parent
        inst_file = root_dir / "instructions" / "instructions (proofread).txt"
        if inst_file.exists():
            settings["instruction_file"] = str(inst_file)
            console.print(
                f"  [green]Proofreading mode enabled[/green] (using {inst_file.name})"
            )
        else:
            logger.warning(f"Proofread instructions not found at {inst_file}")

        # Override output language code to 'proofread' to avoid overwriting original/translated files
        lang_code = "proofread"

    desired_path = _build_translated_output_path(sub_file, lang_code)
    project.InitialiseProject(str(sub_file), str(desired_path))
    project.UpdateProjectSettings(options)

    # Batch subtitles
    from PySubtrans import batch_subtitles

    batch_subtitles(
        project.subtitles,
        scene_threshold=scene_threshold,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    total_batches = (
        sum(len(scene.batches) for scene in project.subtitles.scenes)
        if project.subtitles and project.subtitles.scenes
        else 0
    )
    total_lines = project.subtitles.linecount if project.subtitles else 0

    translator = init_translator(options)
    metrics = (
        TranslationMetrics()
        if (show_metrics and mode == TranslationMode.GEMINI)
        else None
    )
    progress = TranslationProgress() if show_progress else None
    start = time.perf_counter()
    try:
        if metrics:
            metrics.attach(translator)
        if progress:
            progress.attach(translator, sub_file, total_lines)
        project.TranslateSubtitles(translator)
    finally:
        if progress:
            progress.detach(translator, final=True)
        if metrics:
            metrics.detach(translator)
            metrics.render(
                time.perf_counter() - start, total_lines, total_batches, rate_limit
            )
    # Normalise file name and run ssync if we have a file
    normalised = _normalise_translated_output(
        sub_file, desired_path, target_language, lang_code
    )
    if normalised.exists():
        if lang_code == "fi":
            _fix_finnish_subtitles(normalised)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate an .srt (or supported) subtitle file using tuned exsubs defaults"
    )
    parser.add_argument("file", help="Subtitle file to translate (.srt, .ass, .vtt)")
    # provider choice similar to exsubs
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gemini", action="store_true", help="Use Gemini model (default)"
    )
    mode_group.add_argument("--gpt", action="store_true", help="Use ChatGPT model")
    mode_group.add_argument("--claude", action="store_true", help="Use Claude model")
    mode_group.add_argument(
        "--deepseek", action="store_true", help="Use DeepSeek model"
    )
    parser.add_argument(
        "-l", "--language", help="Target language (default from config)"
    )
    parser.add_argument(
        "--proofread",
        action="store_true",
        help="Proofread mode: fix flow and grammar without translating",
    )
    parser.add_argument(
        "--setup-vertex",
        action="store_true",
        help="Run Vertex setup assistant (delegates to exsubs --setup-vertex)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Answer yes to prompts for setup assistant",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable translation progress line"
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not print end-of-run translation metrics",
    )

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

    env_mode = _env_default_mode()
    provider_flag = "gemini"
    if env_mode == TranslationMode.CHATGPT:
        provider_flag = "gpt"
    elif env_mode == TranslationMode.CLAUDE:
        provider_flag = "claude"
    elif env_mode == TranslationMode.DEEPSEEK:
        provider_flag = "deepseek"

    if args.gpt:
        provider_flag = "gpt"
    elif args.claude:
        provider_flag = "claude"
    elif args.deepseek:
        provider_flag = "deepseek"

    try:
        translate_srt_file(
            sub_path,
            args.language,
            provider_flag,
            show_metrics=not args.no_metrics,
            proofread=args.proofread,
        )
        return 0
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
