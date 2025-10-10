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
    """Main entry point for gemini-subtrans command"""
    check_required_imports(['PySubtrans', 'google.genai', 'google.api_core'], 'gemini')

    # Provider configuration
    provider = "Gemini"
    default_model = os.getenv('GEMINI_MODEL') or "Gemini 2.0 Flash"

    parser = CreateArgParser(f"Translates subtitles using a Google Gemini model")
    parser.add_argument('-k', '--apikey', type=str, default=None, help=f"Your Gemini API Key (https://makersuite.google.com/app/apikey)")
    parser.add_argument('-m', '--model', type=str, default=None, help="The model to use for translation")
    args = parser.parse_args()

    logger_options = InitLogger("gemini-subtrans", args.debug)

    try:
        options : Options = CreateOptions(args, provider, model=args.model or default_model)

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
