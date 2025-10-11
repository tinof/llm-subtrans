# LLM-Subtrans Development Guide (CLI Edition)

**Note:** This is a CLI-only fork for Linux/macOS. All GUI and Windows-specific components have been removed.

Project uses Python 3.10+. NEVER import or use deprecated typing members like List, Union or Iterator.

Secrets are stored in a .env file - NEVER read the contents of the file.

Run tests/unit_tests.py at the end of a task to validate the change. Activate the envsubtrans virtual environment first.

## Commands
- **IMPORTANT**: Always use the virtual environment Python: `./envsubtrans/bin/python`
- Run all unit tests: `./envsubtrans/bin/python tests/unit_tests.py`
- Run single test: `./envsubtrans/bin/python -m unittest PySubtrans.UnitTests.test_MODULE`
- Run full test suite: `./envsubtrans/bin/python scripts/run_tests.py`
- Build distribution: `./scripts/makedistro.sh` (or `./scripts/makedistro-mac.sh` for macOS)
- Create virtual environment, install dependencies and configure project: `./install.sh`

## Installation
- **Recommended for users**: `pipx install "llm-subtrans[openai,gemini,claude]"` - Creates isolated environment with CLI commands
- **With MKV extraction support**: `pipx install "llm-subtrans[mkv,openai,gemini,claude]"` - Includes exsubs for extracting and translating MKV subtitles
- **For development**: `./install.sh` or `pip install -e ".[openai,gemini,claude]"` in a venv
- **CLI entry points**: Scripts in `scripts/` use underscores (e.g., `gpt_subtrans.py`) but CLI commands use hyphens (e.g., `gpt-subtrans`)
- **MKV extraction**: Requires mkvtoolnix package (provides mkvmerge and mkvextract commands)

## Code Style

**🚨 CRITICAL RULE: NEVER EVER add imports in the middle of functions or methods - ALWAYS place ALL imports at the top of the file. This is the most important rule in this project - if you violate it you will be fired and replaced by Grok!!!**

- **Naming**: PascalCase for classes and methods, snake_case for variables
- **Imports**: Standard lib → third-party → local, alphabetical within groups
- **Class structure**: Docstring → constants → init → properties → public methods → private methods
- **Type Hints**: Use type hints for parameters, return values, and class variables
  - NEVER put spaces around the `|` in type unions. Use `str|None`, never `str | None`
  - ALWAYS put spaces around the colon introducing a type hint:
  - Examples: 
    `def func(self, param : str) -> str|None:` ✅ 
    `def func(self, param: str) -> str | None:` ❌
- **Docstrings**: Triple-quoted concise descriptions for classes and methods
- **Error handling**: Custom exceptions, specific except blocks, input validation, logging.warning/error
  - User-facing error messages should be localizable, using _()
- **Threading safety**: Use locks (RLock/QRecursiveMutex) for thread-safe operations
  **Regular Expressions**: The project uses the `regex` module for regular expression handling, rather than the standard `re`.
- **Unit Tests**: Extend `LoggedTestCase` from `PySubtrans.Helpers.TestCases` and use `assertLogged*` methods for automatic logging and assertions.
  - **Key Principles**:
    - Prefer `assertLogged*` helper methods over manual logging + standard assertions
    - Use semantic assertions over generic `assertTrue` - the helpers provide `assertLoggedEqual`, `assertLoggedIsNotNone`, `assertLoggedIn`, etc.
    - Include descriptive text as the first parameter to explain what is being tested
    - Optionally provide `input_value` parameter for additional context
  - **Common Patterns**:
    - **Equality**: `self.assertLoggedEqual("field_name", expected, obj.field)`
    - **Type checks**: `self.assertLoggedIsInstance("object type", obj, ExpectedClass)`
    - **None checks**: `self.assertLoggedIsNotNone("result", obj)`
    - **Membership**: `self.assertLoggedIn("key existence", "key", data)`
    - **Comparisons**: `self.assertLoggedGreater("count", actual_count, 0)`
    - **Custom logging**: `self.log_expected_result(expected, actual, description="custom check", input_value=input_data)`
  - **Exception Tests**: Guard with `skip_if_debugger_attached` decorator for debugging compatibility
    - Use `log_input_expected_error(input, ExpectedException, actual_exception)` for exception logging
  - **None Safety**: Use `.get(key, default)` with appropriate default values to avoid Pylance warnings, or assert then test for None values.

## MKV Extraction Feature

The `PySubtrans/MKV` module provides automatic subtitle extraction from MKV files:

- **PySubtrans/MKV/VideoFile.py**: Parses season/episode numbers from filenames for sorting
- **PySubtrans/MKV/Config.py**: Translation mode configuration and language mappings
- **PySubtrans/MKV/SubtitleFilter.py**: Encoding detection, UTF-8 conversion, and subtitle filtering
- **PySubtrans/MKV/MKVExtractor.py**: Intelligent track selection and extraction using mkvmerge/mkvextract
- **PySubtrans/MKV/Diagnostics.py**: System diagnostics for I/O performance testing
- **scripts/exsubs.py**: CLI entry point for extracting and translating MKV subtitles using PySubtrans API

The exsubs script integrates directly with PySubtrans using `Options`, `SubtitleProject`, `batch_subtitles`, and `init_translator`. It supports all translation providers (Gemini, ChatGPT, Claude, DeepSeek) through the `--gemini`, `--gpt`, `--claude`, and `--deepseek` flags.

## Information
Consult `docs/architecture.md` for detailed information on the project architecture and components.
