# LLM-Subtrans (CLI Edition)

**A specialized CLI subtitle translator for Linux and macOS.**

This fork is streamlined for high-performance CLI usage, utilizing modern Python tooling (`uv`) and specialized workflows for power users.

## Key Features

- **CLI Only**: No GUI, no Windows support. Optimized for Linux/macOS terminals.
- **Modern Stack**: Built with [`uv`](https://docs.astral.sh/uv/) for fast, reproducible environments.
- **Specialized Scripts**:
  - `exsubs`: Full pipeline to extract subtitles from MKV, translate, and manage the workflow.
  - `transubs`: Efficient SRT/ASS translation workflow.
- **Gemini 1M Context**: First-class support for Google Gemini's massive context window, enabling whole-file translation for superior context retention.
- **Provider Agnostic**: Support for OpenAI, Google Gemini (Vertex AI & AI Studio), Anthropic Claude, DeepSeek, Mistral, and OpenRouter.

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

**Defaults**: Uses Vertex AI Gemini 2.5 Pro. Optimized for 1M context window. Automatically enables parallel translation for modern Gemini models.

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
export GEMINI_USE_VERTEX=true       # Use Vertex AI (default for exsubs)
export GEMINI_MODEL=gemini-2.5-pro  # Specific model
export SCENE_THRESHOLD=300          # Merge lines closer than 300s (for large context)
export MAX_BATCH_SIZE=600           # Max lines per batch
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
