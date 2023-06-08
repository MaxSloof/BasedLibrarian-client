import yaml

from langchain.llms import VertexAI, OpenAI

from providers.provider_options import ProviderOptions

class LLMProvider():
    """
    Interface for selecting the Generative Language Model Provider.
    """

    @staticmethod
    def _retrieve_config() -> dict:
        _config = './configuration.yaml'

        try:
            with open(_config) as f:
                config = yaml.safe_load(f)
        except:
            raise FileNotFoundError(f'Configuration file not found at {_config}')
        
        return config['generative_model']

    def _select_provider(self):
        env_provider = self._env_provider.lower()
        
        # Set LLM Provider
        if env_provider == ProviderOptions.BasedLibrarianServer.value:
            raise NotImplementedError
        elif env_provider == ProviderOptions.LlamaCPP.value:
            raise NotImplementedError
        elif env_provider == ProviderOptions.VertexAI.value:
            provider = VertexAI(**self.config)
        elif env_provider == ProviderOptions.OpenAI.value:
            provider = OpenAI(**self.config)
        elif KeyError:
            raise KeyError(
                f'''Provider not found or incorrect in confuration.yaml under generative_model!
                The value can be any of the following values: {ProviderOptions.list_enums()}'''
                )
        return provider

    def __init__(self) -> None:
        self.config = self._retrieve_config()
        self._env_provider = self.config.pop('provider')
        self.provider = self._select_provider()
        

    

 