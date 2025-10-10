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
    """Main entry point for llm-subtrans command"""
    check_required_imports(['PySubtrans'])

    # Parse command line arguments
    parser = CreateArgParser("Translates subtitles using OpenRouter or a custom AI model server")
    parser.add_argument('-s', '--server', type=str, default=None, help="Address of the server including port (e.g. http://localhost:1234). If not specified, uses OpenRouter")
    parser.add_argument('-e', '--endpoint', type=str, default=None, help="Endpoint to call on the server (e.g. /v1/completions)")
    parser.add_argument('-k', '--apikey', type=str, default=None, help="API Key (if required)")
    parser.add_argument('-m', '--model', type=str, default=None, help="Model to use if the server allows it to be specified")
    parser.add_argument('--auto', action='store_true', help="Use OpenRouter's automatic model selection")
    parser.add_argument('--chat', action='store_true', help="Use chat format requests for the endpoint")
    parser.add_argument('--systemmessages', action='store_true', help="Indicates that the endpoint supports system messages in chat requests")
    args = parser.parse_args()

    # Determine provider based on whether server is specified
    provider = "Custom Server" if args.server else "OpenRouter"

    logger_options = InitLogger("llm-subtrans", args.debug)

    try:
        if provider == "OpenRouter":
            options : Options = CreateOptions(
                args,
                provider,
                api_key=args.apikey,
                model=args.model,
                use_default_model=args.auto
            )
        else:
            options : Options = CreateOptions(
                args,
                provider,
                api_key=args.apikey,
                endpoint=args.endpoint,
                model=args.model,
                server_address=args.server,
                supports_conversation=args.chat,
                supports_system_messages=args.systemmessages
            )

        # Create a project for the translation
        project : SubtitleProject = CreateProject(options, args)

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
