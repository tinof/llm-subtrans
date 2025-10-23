import importlib.util
import logging
import os

from PySubtrans.Options import SettingsType, env_float, env_int
from PySubtrans.SettingsType import GuiSettingsType, SettingsType

if not importlib.util.find_spec("google"):
    from PySubtrans.Helpers.Localization import _
    logging.debug(_("Google SDK (google-genai) is not installed. Gemini provider will not be available"))
else:
    try:
        from collections import defaultdict

        from google import genai
        from google.genai.types import ListModelsConfig
        from google.api_core.exceptions import FailedPrecondition

        from PySubtrans.Helpers.Localization import _
        from PySubtrans.Providers.Clients.GeminiClient import GeminiClient
        from PySubtrans.TranslationClient import TranslationClient
        from PySubtrans.TranslationProvider import TranslationProvider

        class GeminiProvider(TranslationProvider):
            name = "Gemini"

            information = """
            <p>Select the <a href="https://ai.google.dev/models/gemini">AI model</a> to use as a translator.</p>
            <p>Please note that the Gemini API can currently only be accessed from IP addresses in <a href="https://ai.google.dev/available_regions">certain regions</a>.</p>
            <p>You must ensure that the Generative Language API is enabled for your project and/or API key.</p>
            """

            information_noapikey = """
            <p>Please note that the Gemini API can currently only be accessed from IP addresses in <a href="https://ai.google.dev/available_regions">certain regions</a>.</p>
            <p>To use this provider you need to create an API Key <a href="https://aistudio.google.com/app/apikey">Google AI Studio</a>
            or a project on <a href="https://console.cloud.google.com/">Google Cloud Platform</a> and enable Generative Language API access.</p>
            """

            def __init__(self, settings : SettingsType):
                super().__init__(self.name, SettingsType({
                    "api_key": settings.get_str('api_key') or os.getenv('GEMINI_API_KEY'),
                    "model": settings.get_str('model') or os.getenv('GEMINI_MODEL'),
                    'stream_responses': settings.get_bool('stream_responses', os.getenv('GEMINI_STREAM_RESPONSES', "True") == "True"),
                    'enable_thinking': settings.get_bool('enable_thinking', os.getenv('GEMINI_ENABLE_THINKING', "False") == "True"),
                    'thinking_budget': settings.get_int('thinking_budget', env_int('GEMINI_THINKING_BUDGET', 100)) or 100,
                    'temperature': settings.get_float('temperature', env_float('GEMINI_TEMPERATURE', 0.0)),
                    'rate_limit': settings.get_float('rate_limit', env_float('GEMINI_RATE_LIMIT', 60.0)),
                    'use_vertex': settings.get_bool('use_vertex', os.getenv('GEMINI_USE_VERTEX', "False") == "True"),
                    'vertex_project': settings.get_str('vertex_project') or os.getenv('VERTEX_PROJECT') or os.getenv('GEMINI_VERTEX_PROJECT'),
                    'vertex_location': settings.get_str('vertex_location') or os.getenv('VERTEX_LOCATION') or os.getenv('GEMINI_VERTEX_LOCATION') or 'europe-west1'
                }))

                self.refresh_when_changed = ['api_key', 'model', 'enable_thinking', 'use_vertex', 'vertex_project', 'vertex_location']
                self.gemini_models = []

            @property
            def api_key(self) -> str|None:
                return self.settings.get_str( 'api_key')

            @property
            def use_vertex(self) -> bool:
                return self.settings.get_bool('use_vertex', False)

            @property
            def vertex_project(self) -> str|None:
                return self.settings.get_str('vertex_project')

            @property
            def vertex_location(self) -> str|None:
                return self.settings.get_str('vertex_location')

            def GetTranslationClient(self, settings : SettingsType) -> TranslationClient:
                client_settings = SettingsType(self.settings.copy())
                client_settings.update(settings)
                client_settings.update({
                    'model': self._get_true_name(self.selected_model),
                    'supports_streaming': True,
                    'supports_conversation': False,         # Actually it does support conversation
                    'supports_system_messages': False,       # This is what it doesn't support
                    'supports_system_prompt': True,
                    'use_vertex': self.use_vertex,
                    'vertex_project': self.vertex_project,
                    'vertex_location': self.vertex_location
                    })
                return GeminiClient(client_settings)

            def GetOptions(self, settings : SettingsType) -> GuiSettingsType:
                options : GuiSettingsType = {
                    'use_vertex': (bool, _("Use Vertex AI (requires Application Default Credentials and Vertex AI enabled in the selected project)"))
                }

                if not self.use_vertex:
                    options['api_key'] = (str, _("A Google Gemini API key is required to use this provider (https://makersuite.google.com/app/apikey)"))

                if self.use_vertex:
                    options['vertex_project'] = (str, _("Google Cloud project ID for Vertex AI requests"))
                    options['vertex_location'] = (str, _("Vertex AI region (e.g. us-central1)"))

                if self.use_vertex or self.api_key:
                    try:
                        models = self.available_models
                        if models:
                            options.update({
                                'model': (models, "AI model to use as the translator" if models else "Unable to retrieve models"),
                                'stream_responses': (bool, _("Stream translations in realtime as they are generated")),
                                'enable_thinking': (bool, _("Enable reasoning capabilities for more complex translations (increases cost)")),
                                'temperature': (float, _("Amount of random variance to add to translations. Generally speaking, none is best")),
                                'rate_limit': (float, _("Maximum API requests per minute."))
                            })

                            if self.settings.get_bool('enable_thinking', False):
                                options['thinking_budget'] = (int, _("Token budget for reasoning. Higher values increase cost"))

                        else:
                            options['model'] = (["Unable to retrieve models"], _("Check API key is authorized and try again"))

                    except FailedPrecondition as e:
                        options['model'] = (["Unable to access the Gemini API"], str(e))

                return options

            def GetAvailableModels(self) -> list[str]:
                if not self.gemini_models:
                    self.gemini_models = self._get_gemini_models()

                return sorted([m.display_name for m in self.gemini_models])

            def GetInformation(self) -> str:
                return self.information if self.api_key else self.information_noapikey

            def ValidateSettings(self) -> bool:
                """
                Validate the settings for the provider
                """
                if not self.use_vertex and not self.api_key:
                    self.validation_message = _("API Key is required")
                    return False

                if self.use_vertex:
                    if not self.vertex_project:
                        self.validation_message = _("Vertex AI project ID is required")
                        return False

                    if not self.vertex_location:
                        self.validation_message = _("Vertex AI location is required")
                        return False

                # Some regions do not list Gemini models even when generation works.
                # Treat model listing failures as non-fatal for CLI use.
                if not self.GetAvailableModels():
                    logging.warning(_("Unable to retrieve Gemini model list; proceeding with configured model"))
                    return True

                return True

            def _get_gemini_models(self):
                try:
                    if not self.use_vertex and not self.api_key:
                        return []

                    gemini_client = self._create_client()
                    config = ListModelsConfig(query_base=True) if not self.use_vertex else ListModelsConfig()
                    all_models = gemini_client.models.list(config=config)

                    if self.use_vertex:
                        vertex_models = []
                        for model in all_models:
                            model_name = getattr(model, 'name', '') or ''
                            if 'models/gemini' not in model_name:
                                continue

                            if 'embedding' in model_name:
                                continue

                            if getattr(model, 'display_name', None) is None:
                                try:
                                    model.display_name = model_name.split('/')[-1]
                                except Exception:
                                    model.display_name = model_name

                            vertex_models.append(model)

                        return sorted({m.display_name: m for m in vertex_models}.values(), key=lambda m: m.display_name)

                    generate_models = [ m for m in all_models if m.supported_actions and 'generateContent' in m.supported_actions ]
                    text_models = [m for m in generate_models if m.display_name and "Vision" not in m.display_name and "TTS" not in m.display_name]

                    return self._deduplicate_models(text_models)

                except Exception as e:
                    logging.error(_("Unable to retrieve Gemini model list: {error}").format(error=str(e)))
                    return []

            def _get_true_name(self, name : str|None) -> str:
                if not self.gemini_models:
                    self.gemini_models = self._get_gemini_models()

                if not name:
                    return self.gemini_models[0].name if self.gemini_models else ""

                for m in self.gemini_models:
                    if m.display_name == name:
                        return m.name

                    if name and m.name.endswith(name):
                        return m.name

                    if m.name == f"models/{name}" or m.name == name:
                        return m.name

                # Fallback: trust the provided name if the model list is unavailable
                return name or ""

            def _create_client(self):
                if self.use_vertex:
                    if not self.vertex_project:
                        raise ValueError(_("Vertex AI project ID not configured"))

                    return genai.Client(vertexai=True, project=self.vertex_project, location=self.vertex_location)

                return genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})

            def _deduplicate_models(self, models : list) -> list:
                """Deduplicate models by display name, preferring -latest versions"""
                # Group models by display name
                model_groups = defaultdict(list)
                for model in models:
                    model_groups[model.display_name].append(model)
                
                # Select best model for each display name  
                selected_models = [
                    latest_models[0] if (latest_models := [m for m in models if m.name.endswith('-latest')])
                    else min(models, key=lambda m: len(m.name))
                    for models in model_groups.values()
                ]
                
                return selected_models

            def _allow_multithreaded_translation(self) -> bool:
                """
                If user has set a rate limit don't attempt parallel requests to make sure we respect it
                """
                if self.settings.get_float( 'rate_limit', 0.0) != 0.0:
                    return False

                return True

    except ImportError:
        from PySubtrans.Helpers.Localization import _
        logging.info(_("Latest Google AI SDK (google-genai) is not installed. Gemini provider will not be available. Run installer or `pip install google-genai` to fix."))
