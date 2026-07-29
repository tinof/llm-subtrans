# LLM-Subtrans (CLI Edition)

A specialized CLI subtitle translator for Linux and macOS. 

This repository is a hard fork of [machinewrapped/llm-subtrans](https://github.com/machinewrapped/llm-subtrans), stripped of all Windows/GUI bloat and heavily optimized for terminal power users, automation, and superior linguistic quality.

## Key Features & Upgrades

- **Smart Line Merging (`[MERGE]` tag):** The upstream engine strictly enforces a 1:1 translation ratio per line. This fork allows the AI to merge fragmented source lines into a single fluent line—critical for agglutinative languages (like Finnish, Turkish, or Japanese) where word order completely changes.
- **All-in-One MKV Pipeline (`exsubs`):** Bypasses the need for external tools. Automatically extracts subtitle tracks directly from `.mkv` files, filters them, translates them, and saves the output.
- **Parallel Translation:** `--parallel` mode translates multiple subtitle batches simultaneously via `ThreadPoolExecutor` (auto-enabled with 8 workers for modern Gemini models), making large jobs up to 8x faster.
- **Large Context Mode:** Built to utilize the massive context windows of modern models (like Gemini 2.5/3.x). It feeds the AI thousands of tokens of story history to drastically improve character consistency and tone over long movies.
- **Resilient API Handling:** Transient provider errors (429/5xx) are retried with jittered exponential backoff; on Vertex AI (dynamic shared quota) no artificial client-side RPM limit is applied.
- **Readability-Aware Translation:** Each line in the prompt carries its display time and a character budget (`#123 [2.4s, max 36 chars]`, derived from a 15 CPS target), so the model condenses lines that would be unreadable instead of translating word-for-word. The Finnish instructions also have the model infer dialogue dashes for two-speaker cues — sources like Channel 4 ship none.
- **CLI-Native Modern Stack:** No PySide6, no Windows hooks, no PyInstaller scripts. Built entirely around [`uv`](https://docs.astral.sh/uv/) for incredibly fast dependency and environment management.
- **Direct Subtitle Workflow (`transubs`):** Efficient, highly-tuned SRT/ASS/VTT translation script for when you don't need MKV extraction.
  
## Installation

Install using [uv](https://docs.astral.sh/uv/) (recommended):

```sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install llm-subtrans with common providers
uv tool install "llm-subtrans[openai,gemini,claude] @ git+https://github.com/tinof/llm-subtrans.git"

# To update
uv tool upgrade llm-subtrans
```

## Workflows

### 1. `exsubs` — Extract & Translate from MKV

The primary workflow for video files. Extracts subtitles from MKV and translates them using optimized defaults.

```sh
# Translate MKV subtitles to Finnish using Gemini (Vertex AI default)
exsubs video.mkv --gemini -l Finnish

# Process all MKV files in the current directory
exsubs --gemini -l Finnish

# Interactive track selection
exsubs video.mkv --gpt -l Spanish -i
```

**Defaults**: Uses Gemini via Vertex AI (default model: `gemini-3.1-flash-lite`, override with `GEMINI_MODEL`). Optimized for 1M context window. Automatically enables parallel translation for modern Gemini models. Temperature defaults to 1.0 on Gemini 3.x (lowering it causes repetition loops) and 0.3 elsewhere; override with `LLM_TEMPERATURE`.

**Finnish post-processing**: after translation, `exsubs` runs
[sisusub](https://github.com/tinof/sisusub)'s `fix-finnish-subs` once with
`--width-limit 42 --max-cps 17 --cps-target 15` (deterministic layout/timing
fixes plus an AI review pass; set `SISUSUB_AI_MODEL` to run the review on a
stronger model than the translator). A readability report is written next to
the output; set `EXSUBS_FIXER_VERBOSE=1` for per-proposal logging. Measure any
result objectively with:

```sh
uv run python tools/subtitle_metrics.py translated.fi.srt --source english.srt
```

#### `exsubs` Options

| Flag                  | Description                                                                    |
| --------------------- | ------------------------------------------------------------------------------ |
| `--gemini`            | Use Gemini model (default)                                                     |
| `--gpt`               | Use OpenAI GPT model                                                           |
| `--claude`            | Use Anthropic Claude model                                                     |
| `--deepseek`          | Use DeepSeek model                                                             |
| `-l`, `--language`    | Target language (e.g., `Finnish`, `Spanish`)                                   |
| `-i`, `--interactive` | Manually select subtitle track to extract                                      |
| `--no-filter`         | Preserve music cues, fonts, effects, etc. (disabled by default)                |
| `--no-progress`       | Hide progress bars                                                             |
| `--no-metrics`        | Don't print translation statistics at the end                                  |
| `--parallel`          | Enable parallel batch translation (auto-enabled for Gemini Flash/Pro)          |
| `--no-parallel`       | Force sequential translation                                                   |
| `--parallel-workers`  | Number of parallel workers (default: 4, or 8 for modern Gemini)                |
| `--max-batch-size`    | Max lines per batch (default: 600 for Gemini, 200 for parallel)                |
| `--min-batch-size`    | Min lines per batch                                                            |
| `--copy-local`        | Copy MKV to local temp before processing (useful for network mounts)           |
| `--setup-vertex`      | Interactive wizard to configure Vertex AI                                      |
| `--diagnose`          | Run system diagnostics                                                         |

---

### 2. `transubs` — Subtitle File Translation

Directly translate existing subtitle files (.srt, .ass, .vtt) with the same tuned engine as `exsubs`.

```sh
# Translate an SRT file
transubs input.srt --gemini -l French

# Translate without removing music cues, fonts, effects, etc.
transubs --no-filter input.srt -l German
```

#### `transubs` Options

| Flag               | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `--gemini`         | Use Gemini model (default)                            |
| `--gpt`            | Use OpenAI GPT model                                  |
| `--claude`         | Use Anthropic Claude model                            |
| `--deepseek`       | Use DeepSeek model                                    |
| `-l`, `--language` | Target language                                       |
| `--proofread`      | Fix flow/grammar without translating                  |
| `--no-filter`      | Preserve music cues, fonts, effects, etc.             |
| `--no-progress`    | Hide translation progress line                        |
| `--no-metrics`     | Don't print translation statistics at the end         |
| `--setup-vertex`   | Interactive wizard to configure Vertex AI             |

---

### 3. `llm-subtrans` — Universal Tool

The base tool for fine-grained control, legacy compatibility, or specific provider usage not covered by the wrapper scripts. Run `llm-subtrans --help` for all options.

```sh
# Auto-select model via OpenRouter
llm-subtrans --auto -l Japanese subtitle.srt
```

---

## Configuration

Settings are managed via environment variables or `.env` file. Common overrides:

```sh
export GEMINI_USE_VERTEX=true             # Use Vertex AI (default for exsubs)
export GEMINI_MODEL=gemini-3.1-flash-lite # Specific model
export SCENE_THRESHOLD=300                # Start a new scene when the gap between lines exceeds 300s (large context)
export MAX_BATCH_SIZE=600                 # Max lines per batch
export MAX_RETRIES=5                      # Retries for transient API errors (default: 5)
export LLM_TEMPERATURE=0.3                # Override the model-aware temperature default
export SISUSUB_AI_MODEL=gemini-3.1-pro-preview  # Stronger model for the fix-finnish-subs review pass
export EXSUBS_FIXER_VERBOSE=1             # Verbose fix-finnish-subs output (dropped-proposal logging)
```

**Vertex AI Setup**: Run `exsubs --setup-vertex` for a guided configuration wizard.

---

## Development

This project uses `uv` for all development tasks.

```sh
git clone https://github.com/tinof/llm-subtrans.git
cd llm-subtrans
uv sync --all-extras --dev

# Run tests
uv run pytest

# Format/Lint
uv run ruff check --fix && uv run ruff format
```

---

## License

MIT License.
