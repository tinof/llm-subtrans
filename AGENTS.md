# LLM-Subtrans Development Guide (CLI Edition)

**Note:** This is a CLI-only fork for Linux/macOS. All GUI and Windows-specific components have been removed.

Project uses Python 3.10+. NEVER import or use deprecated typing members like List, Union or Iterator.

Secrets are stored in a .env file - NEVER read the contents of the file.

## Package Management (STRICT)

- STRICTLY use `uv` for dependency management, environment management, and command execution.
- Do not use `pip`, `pipx`, `pipxu`, or manual virtual environment activation flows for normal project work.
- Use `uv sync` to install dependencies and `uv run` to execute project commands.
- Use `uv tool install` to install CLI commands system-wide.

## Installation

### Development (in the repo checkout)

```bash
uv sync --all-extras --dev
```

This installs all optional provider dependencies (Gemini, OpenAI, Claude, etc.) and dev tools (ruff, basedpyright, pytest).

### System-Wide CLI (via `uv tool`)

Install the CLI commands (`exsubs`, `transubs`, etc.) globally:

```bash
uv tool install "llm-subtrans[openai,gemini,claude]" @ git+https://github.com/tinof/llm-subtrans.git
```

Or for editable/local development:

```bash
uv tool install --editable /path/to/llm-subtrans --with "google-genai" --with "google-api-core" --with "openai" --with "anthropic"
```

### Optional Extras

Provider-specific dependencies are gated behind extras in `pyproject.toml`:

| Extra | Dependencies | Required For |
| --- | --- | --- |
| `openai` | `openai` | GPT, Azure providers |
| `gemini` | `google-genai`, `google-api-core` | Gemini / Vertex AI provider |
| `claude` | `anthropic` | Claude provider |
| `mistral` | `mistralai` | Mistral provider |
| `bedrock` | `boto3` | AWS Bedrock provider |
| `mkv` | `subtitle_filter`, `chardet`, `filelock`, `rich` | MKV subtitle extraction |

**If a provider extra is not installed, the provider silently won't register and you'll get `Unknown translation provider` errors at runtime.**

## Command Reference

| Task | Command |
| --- | --- |
| Install deps (dev) | `uv sync --all-extras --dev` |
| Format | `uv run ruff check --fix && uv run ruff format` |
| Checks | `uv run ruff check && uv run ruff format --check && uv run basedpyright` |
| Tests | `uv run pytest` |

### Makefile Targets

All of the above are also available as `make` targets:

| Target | Command | Description |
| --- | --- | --- |
| `make` (default) | `install check test` | Full pipeline: install, lint, test |
| `make install` | `uv sync --all-extras` | Install all deps including all provider extras |
| `make fmt` | ruff check --fix + ruff format | Auto-fix lint issues and reformat |
| `make check` | ruff check + format --check + basedpyright | Lint, format verify, and type check |
| `make test` | `uv run pytest` | Run test suite |
| `make build` | `uv build` | Build distribution package |
| `make clean` | rm -rf dist, caches, venv | Clean all artifacts |

## Project Structure Overview

- `PySubtrans/`: Core subtitle translation engine, providers, parsing, and helpers.
  - `Providers/`: Pluggable translation provider implementations (Gemini, OpenAI, Claude, etc.).
  - `Formats/`: Subtitle format handlers (SRT, ASS/SSA, VTT) with auto-discovery registry.
  - `Helpers/`: Shared utilities, test base classes (`LoggedTestCase`, `DummyProvider`).
- `scripts/`: CLI entrypoints and supporting automation scripts.
  - `exsubs.py`: Extract subtitles from MKV files and translate (primary workflow).
  - `transubs.py`: Translate existing subtitle files (.srt, .ass, .vtt).
  - `batch_translate.py`: Batch translation of entire directories.
  - `subtrans_common.py`: Shared CLI infrastructure (arg parsing, options, project creation).
  - `check_imports.py`: Validates provider dependencies are importable before running.
  - Provider-specific scripts: `gemini_subtrans.py`, `gpt_subtrans.py`, `claude_subtrans.py`, etc.
- `tests/`: Unit test suites and test harnesses.
  - `PySubtransTests/`: Core unit tests extending `LoggedTestCase`.
  - `functional/`: Functional tests (preprocessor, batcher).
  - `TestData/`: Realistic test data fixtures.
- `instructions/`: Translation/system instruction templates shipped with the project.
- `docs/`: Architecture and contributor documentation.
- `.github/workflows/`: CI and automation workflows.

## Code Style Guidelines

**CRITICAL RULE: NEVER add imports in the middle of functions or methods - ALL imports MUST be at the top of the file.**

- **Naming**: PascalCase for classes and methods, snake_case for variables
- **Imports**: Standard lib -> third-party -> local, alphabetical within groups
- **Class structure**: Docstring -> constants -> init -> properties -> public methods -> private methods
- **Type Hints**: Use type hints for parameters, return values, and class variables
  - Follow PEP 8/604 style: no space before colon, space after colon, spaces around `|`
  - Example: `def func(self, param: str) -> str | None:`
- **Docstrings**: Triple-quoted concise descriptions for classes and methods
- **Error handling**: Custom exceptions, specific except blocks, input validation, logging.warning/error
  - User-facing error messages should be localizable, using `_()`
- **Threading safety**: Use locks (RLock/QRecursiveMutex) for thread-safe operations
- **Regular Expressions**: Use the `regex` module rather than standard `re`
- **Unit Tests**: Extend `LoggedTestCase` from `PySubtrans.Helpers.TestCases` and prefer `assertLogged*` methods

## Testing and Validation

- Always run `make check` and `make test` before finishing a task.
- If needed for compatibility validation, run `uv run python tests/unit_tests.py`.

## Information

Consult `docs/architecture.md` for detailed information on the project architecture and components.

## Upstream Sync (Fork Maintenance)

- `upstream` is the Git remote pointing to the original repo (verify with `git remote -v`). In this fork it is set to `https://github.com/machinewrapped/llm-subtrans.git`.
- Preferred workflow for public main: merge (not rebase) to preserve history and avoid force-pushes.
- Keep our CLI-only customizations intact: do not reintroduce GUI or Windows-specific code.

### Quick Steps

- Fetch upstream: `git fetch upstream`
- Review ahead/behind:
  - Upstream ahead: `git log --oneline main..upstream/main`
  - Fork ahead: `git log --oneline upstream/main..main`
- Merge: `git merge upstream/main` and resolve conflicts by
  - Preserving: top-level `pyproject.toml` (CLI description/version, scripts) and Linux/macOS-only constraints
  - Adopting: changes under `PySubtrans/` (core engine), tests, locales
- Validate after merge: `make check && make test`
- Push if green: `git push origin main`

### Prompt Template (for Codex)

- "Merge upstream/main into our main; preserve CLI-only fork constraints (no GUI/Windows), keep top-level pyproject CLI metadata/scripts, adopt upstream PySubtrans changes, run checks/tests with make, and push to origin/main if tests pass."

### When to Rebase

- Only if a linear history is required and force-push is acceptable. Otherwise, prefer merges.

## Repository Hygiene

- Never print or read `.env` contents.
- New files or scripts must not add imports in the middle of functions.
- Use `regex` (not `re`) for regular expressions.
