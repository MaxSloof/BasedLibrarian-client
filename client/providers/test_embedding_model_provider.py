import unittest
from unittest.mock import patch
from providers.embedding_model_provider import EmbeddingModelProvider
from providers.provider_options import ProviderOptions
from langchain.embeddings import OpenAIEmbeddings
from google.auth.exceptions import GoogleAuthError


class TestGenerativeModelProvider(unittest.TestCase):

    def test_find_config_file(self):
        config = EmbeddingModelProvider._retrieve_config()
        self.assertEqual(type(config), dict)

    @patch('providers.embedding_model_provider.yaml.safe_load')
    def test_retrieve_config_success(self, mock_safe_load):
        mock_safe_load.return_value = {
            'embedding_model': {
                'provider': ProviderOptions.VertexAI.value,
                'temperature': 0.7
                }
                }
        config = EmbeddingModelProvider._retrieve_config()

        self.assertEqual(config, {'provider': ProviderOptions.VertexAI.value, 'temperature': 0.7})

    @patch('providers.embedding_model_provider.yaml.safe_load')
    def test_retrieve_config_file_not_found(self, mock_safe_load):
        mock_safe_load.side_effect = FileNotFoundError('Configuration file not found')
        
        with self.assertRaises(FileNotFoundError):
            EmbeddingModelProvider._retrieve_config()

    @patch('providers.embedding_model_provider.EmbeddingModelProvider._retrieve_config')
    def test_select_provider_vertex_ai(self, mock_retrieve_config):
        mock_retrieve_config.return_value = {
            'provider': ProviderOptions.VertexAI.value, 
            'credentials': "placeholder",
            'project': "my-placeholder-project"
        }

        with self.assertRaises(GoogleAuthError):
            EmbeddingModelProvider()

    
    @patch('providers.embedding_model_provider.EmbeddingModelProvider._retrieve_config')
    def test_select_provider_open_ai(self, mock_retrieve_config):
        mock_retrieve_config.return_value = {
            'provider': "OpenAI", 
            "openai_api_key": "placeholder", 
            }
        
        provider = EmbeddingModelProvider()
        selected_provider = provider._select_provider()

        self.assertEqual(type(selected_provider), OpenAIEmbeddings)
    
    
    @patch('providers.embedding_model_provider.EmbeddingModelProvider._retrieve_config')
    def test_select_provider_invalid_provider(self, mock_retrieve_config):
        mock_retrieve_config.return_value = {'provider': 'InvalidProvider'}
        
        with self.assertRaises(KeyError):
            provider = EmbeddingModelProvider()

    @patch('providers.embedding_model_provider.EmbeddingModelProvider._retrieve_config')
    def test_class_returns_openai_class(self, mock_retrieve_config):
        # Tests whether the class returns the correct class based on the config file
        # And whether the params are correctly passed from the config file to the generative model class
        mock_retrieve_config.return_value = {
            'provider': "OpenAI", 
            "openai_api_key": "placeholder", 
            }
        
        generative_model = EmbeddingModelProvider().provider
        self.assertIs(type(generative_model), OpenAIEmbeddings)
        
if __name__ == '__main__':
    unittest.main() # pragma: no cover
