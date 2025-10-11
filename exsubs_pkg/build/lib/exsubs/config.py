#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


class TranslationMode(str, Enum):
    CHATGPT = "chatgpt"  # Set as first/default option
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


# Centralized mappings to reduce duplication
MODE_TO_CMD = {
    TranslationMode.CHATGPT: ["gpt-subtrans"],
    TranslationMode.CLAUDE: ["claude-subtrans"],
    TranslationMode.GEMINI: ["gemini-subtrans"],
    TranslationMode.DEEPSEEK: ["deepseek-subtrans"],
}

MODE_TO_DEFAULT_MODEL = {
    TranslationMode.CHATGPT: "gpt-5-mini",
    TranslationMode.CLAUDE: "claude-3-5-haiku-latest",
    TranslationMode.GEMINI: "gemini-2.5-flash",
    TranslationMode.DEEPSEEK: "deepseek-chat",
}

MODE_TO_ENV = {
    TranslationMode.CHATGPT: "OPENAI_API_KEY",
    TranslationMode.CLAUDE: "CLAUDE_API_KEY",
    TranslationMode.GEMINI: "GEMINI_API_KEY",
    TranslationMode.DEEPSEEK: "DEEPSEEK_API_KEY",
}

MODE_TO_RATE_LIMIT = {
    TranslationMode.CHATGPT: "5000",
    TranslationMode.CLAUDE: "600",
    TranslationMode.GEMINI: "1000",
    TranslationMode.DEEPSEEK: "4000",
}


@dataclass
class Config:
    instruction_file: Path = Path("/home/siteadmin/bin/instructions_fi.txt")
    config_file: Path = Path("/etc/gpt-subtrans/.env")
    target_language: str = "Finnish"
    default_translation_mode: TranslationMode = TranslationMode.GEMINI

    SUPPORTED_LANGUAGES = ["Finnish", "Greek", "German", "French", "Spanish", "Italian"]

    @staticmethod
    def get_language_code(lang: str) -> str:
        codes = {
            "Finnish": "fi",
            "Greek": "el",
            "German": "de",
            "French": "fr",
            "Spanish": "es",
            "Italian": "it",
        }
        return codes.get(lang, lang)

    @staticmethod
    def get_flag(lang: str) -> str:
        flags = {"Finnish": "🇫🇮", "Greek": "🇬🇷"}
        return flags.get(lang, "")
