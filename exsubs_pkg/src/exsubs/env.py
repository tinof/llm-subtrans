#!/usr/bin/env python3

import logging
import os
import subprocess
import sys
import importlib.util

from .config import MODE_TO_ENV, TranslationMode

logger = logging.getLogger("exsubs")


def verify_dependencies():
    """Verify required command line tools and Python packages are available"""
    # Check for command-line tools
    required_commands = ["mkvextract", "mkvmerge", "filter-subtitles.py"]
    for cmd in required_commands:
        if not _command_exists(cmd):
            logger.error(f"{cmd} not found. Please install {cmd}.")
            sys.exit(1)

    # Check for required Python packages
    required_packages = {
        "filelock": "filelock",
        "subtitle_filter": "subtitle-filter",
    }
    for import_name, install_name in required_packages.items():
        if not _package_exists(import_name):
            logger.error(
                f"Python package '{import_name}' not found. Please install it with 'pip install {install_name}'."
            )
            sys.exit(1)

    # Check for optional packages that improve functionality
    optional_packages = {"chardet": "chardet"}
    for import_name, install_name in optional_packages.items():
        if not _package_exists(import_name):
            logger.warning(
                f"Optional package '{import_name}' not found. "
                f"Install it with 'pip install {install_name}' for better encoding detection."
            )


def verify_api_keys(mode: TranslationMode):
    """Verify required API keys are set based on translation mode"""
    required_key = MODE_TO_ENV[mode]
    if not os.getenv(required_key):
        logger.error(f"{required_key} not found in environment")
        sys.exit(1)


def _command_exists(cmd: str) -> bool:
    return subprocess.run(["which", cmd], capture_output=True).returncode == 0


def _package_exists(package_name: str) -> bool:
    """Check if a Python package is installed"""
    return importlib.util.find_spec(package_name) is not None
