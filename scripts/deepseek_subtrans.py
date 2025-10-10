import os
import logging

from scripts.check_imports import check_required_imports
from scripts.subtrans_common import (
    InitLogger,
    CreateArgParser,
    CreateOptions,
    CreateProject,
)

from PySubtrans import init_translator
from PySubtrans.Options import Options
from PySubtrans.SubtitleProject import SubtitleProject

def main() -> int:
    """Main entry point for deepseek-subtrans command"""
    check_required_imports(['PySubtrans'])

    # Provider configuration
    provider = "DeepSeek"
    default_model = os.getenv('DEEPSEEK_MODEL') or "deepseek-chat"

    parser = CreateArgParser(f"Translates subtitles using an DeepSeek model")
    parser.add_argument('-k', '--apikey', type=str, default=None, help=f"Your DeepSeek API Key (https://platform.deepseek.com/api_keys)")
    parser.add_argument('-b', '--apibase', type=str, default="https://api.deepseek.com", help="API backend base address.")
    parser.add_argument('-m', '--model', type=str, default=None, help="The model to use for translation")
    args = parser.parse_args()

    logger_options = InitLogger("deepseek-subtrans", args.debug)

    try:
        options : Options = CreateOptions(
            args,
            provider,
            api_base=args.apibase,
            model=args.model or default_model
        )

        # Create a project for the translation
        project : SubtitleProject = CreateProject(options, args)

        # Translate the subtitles
        translator = init_translator(options)
        project.TranslateSubtitles(translator)

        if project.use_project_file:
            logging.info(f"Writing project data to {str(project.projectfile)}")
            project.SaveProjectFile()

        return 0

    except Exception as e:
        print("Error:", e)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
