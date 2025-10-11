import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class TranslationMode(str, Enum):
    """Supported translation modes"""
    CHATGPT = "chatgpt"
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
    TranslationMode.CHATGPT: "gpt-4o-mini",
    TranslationMode.CLAUDE: "claude-3-5-haiku-latest",
    TranslationMode.GEMINI: "gemini-2.0-flash",
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
class MKVConfig:
    """Configuration for MKV subtitle extraction and translation"""
    instruction_file : Path|None = None
    target_language : str = "Finnish"
    default_translation_mode : TranslationMode = TranslationMode.GEMINI

    SUPPORTED_LANGUAGES = ["Finnish", "Greek", "German", "French", "Spanish", "Italian", "English"]

    def __post_init__(self):
        """Initialize configurable paths from environment or defaults"""
        if self.instruction_file is None:
            # Check environment variable first
            env_path = os.getenv('LLMSUBTRANS_INSTRUCTION_FILE')
            if env_path:
                self.instruction_file = Path(env_path)
            else:
                # Default to user's config directory
                config_dir = Path.home() / '.config' / 'llm-subtrans'
                lang_code = self.get_language_code(self.target_language)
                self.instruction_file = config_dir / f'instructions_{lang_code}.txt'

    @staticmethod
    def get_language_code(lang : str) -> str:
        """Get ISO 639-1 language code from language name"""
        codes = {
            "Finnish": "fi",
            "Greek": "el",
            "German": "de",
            "French": "fr",
            "Spanish": "es",
            "Italian": "it",
            "English": "en",
        }
        return codes.get(lang, lang.lower()[:2])

    @staticmethod
    def get_flag(lang : str) -> str:
        """Get emoji flag for language"""
        flags = {
            "Finnish": "🇫🇮",
            "Greek": "🇬🇷",
            "German": "🇩🇪",
            "French": "🇫🇷",
            "Spanish": "🇪🇸",
            "Italian": "🇮🇹",
            "English": "🇬🇧",
        }
        return flags.get(lang, "")
