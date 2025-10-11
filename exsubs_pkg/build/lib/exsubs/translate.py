#!/usr/bin/env python3

import logging
import subprocess
from pathlib import Path
from typing import List

from .config import (
    Config, TranslationMode, MODE_TO_CMD, MODE_TO_DEFAULT_MODEL, 
    MODE_TO_RATE_LIMIT
)

logger = logging.getLogger("exsubs")


def build_translation_cmd(
    mode: TranslationMode, 
    config: Config, 
    sub_file: Path, 
    out_file: Path, 
    language: str
) -> List[str]:
    """Build the translation command based on mode and config"""
    base_cmd = MODE_TO_CMD[mode]
    
    base_args = [
        "-l", language,
        "-o", str(out_file),
        "--temperature", "0.2",
        "--preprocess",
        "--postprocess",
        "--ratelimit", MODE_TO_RATE_LIMIT[mode],
        "--instructionfile", str(config.instruction_file),
        "--model", MODE_TO_DEFAULT_MODEL[mode],
    ]
    
    return base_cmd + base_args + [str(sub_file)]


def run_translation(cmd: List[str], out_file: Path, cleanup_source: Path = None):
    """Run translation command and cleanup on success"""
    try:
        subprocess.run(cmd, check=True)
        if out_file.exists():
            if cleanup_source and cleanup_source.exists():
                cleanup_source.unlink()
        else:
            logger.error("Translation failed - keeping original subtitle file for debugging")
    except subprocess.CalledProcessError as e:
        logger.error(f"Translation failed: {e}")
        raise
