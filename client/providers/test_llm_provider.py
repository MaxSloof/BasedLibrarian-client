import unittest
from unittest.mock import patch
from client.providers.llm_provider import LLMProvider
from client.providers.provider_options import ProviderOptions
from langchain.llms import OpenAI
from google.auth.exceptions import GoogleAuthError


class TestLLMProvider(unittest.TestCase):

    def test_find_config_file(self):
        config = LLMProvider._retrieve_config()
        self.assertEqual(type(config), dict)

    @patch('client.providers.llm_provider.LLMProvider.yaml.safe_load')
    def test_retrieve_config_success(self, mock_safe_load):
        mock_safe_load.return_value = {
            'generative_model': {
                'provider': ProviderOptions.VertexAI.value,
                'temperature': 0.7,
                }
                }
        config = LLMProvider()._retrieve_config()

        self.assertEqual(config, {'provider': ProviderOptions.VertexAI.value, 'temperature': 0.7})

    @patch('client.providers.llm_provider.LLMProvider.yaml.safe_load')
    def test_retrieve_config_file_not_found(self, mock_safe_load):
        mock_safe_load.side_effect = FileNotFoundError('Configuration file not found')
        
        with self.assertRaises(FileNotFoundError):
            LLMProvider._retrieve_config()

    @patch('client.providers.llm_provider.LLMProvider._retrieve_config')
    def test_select_provider_vertex_ai(self, mock_retrieve_config):
        mock_retrieve_config.return_value = {
            'provider': ProviderOptions.VertexAI.value, 
            'credentials': "placeholder",
            'project': "my-placeholder-project"
        }

        with self.assertRaises(GoogleAuthError):
            LLMProvider()
    
    @patch('client.providers.llm_provider.LLMProvider._retrieve_config')
    def test_select_provider_open_ai(self, mock_retrieve_config):

        mock_retrieve_config.return_value = {
            'provider': "OpenAI", 
            "openai_api_key":"placeholder", 
            "temperature": 0.4
            }
        
        provider = LLMProvider()
        selected_provider = provider._select_provider()

        self.assertEqual(type(selected_provider), OpenAI)
    
    
    @patch('client.providers.llm_provider.LLMProvider._retrieve_config')
    def test_select_provider_invalid_provider(self, mock_retrieve_config):
        mock_retrieve_config.return_value = {'provider': 'InvalidProvider'}
        
        with self.assertRaises(KeyError):
            provider = LLMProvider()

    @patch('client.providers.llm_provider.LLMProvider._retrieve_config')
    def test_class_returns_openai_class(self, mock_retrieve_config):
        # Tests whether the class returns the correct class based on the config file
        # And whether the params are correctly passed from the config file to the generative model class
        mock_retrieve_config.return_value = {
            'provider': "OpenAI", 
            "openai_api_key":"placeholder", 
            "temperature": 0.4
            }
        
        generative_model = LLMProvider().provider
        self.assertIs(type(generative_model), OpenAI)
        self.assertEqual(generative_model.temperature,0.4)
        
if __name__ == '__main__':
    unittest.main()
