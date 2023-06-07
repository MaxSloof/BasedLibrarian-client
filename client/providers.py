import os
from enum import Enum

from langchain.embeddings import VertexAIEmbeddings, OpenAIEmbeddings
from langchain.llms import VertexAI, OpenAI


class ProviderOptions(Enum):
    BasedLibrarianServer = "blserver"
    VertexAI = "vertexai"
    OpenAI = "openai"
    LlamaCPP = "llamacpp"

    @classmethod
    def list_enums(cls):
        return [enum.value for enum in cls]


class Providers():

    def __init__(
            self,
            temperature: float = 0,
            max_output_tokens: int = 256,
            llm_model: str = "NA",
            embeddings_model: str = "NA",
            OpenAIclient=None
            ) -> None:
        ENV_llm_provider = os.environ["LLM_PROVIDER"].lower()

        if os.environ["EMBEDDING_PROVIDER"] == KeyError():
            ENV_embeddings_provider = os.environ["LLM_PROVIDER"]
        else:
            ENV_embeddings_provider = os.environ["EMBEDDING_PROVIDER"]

        ENV_embeddings_provider = ENV_embeddings_provider.lower()

        # Set LLM Provider
        if ENV_llm_provider == ProviderOptions.BasedLibrarianServer:
            raise NotImplementedError
        elif ENV_llm_provider == ProviderOptions.LlamaCPP:
            raise NotImplementedError
        elif ENV_llm_provider == ProviderOptions.VertexAI:
            self.LLM_provider = VertexAI(
                temperature=temperature,
                max_output_tokens=max_output_tokens
                )
        elif ENV_llm_provider == ProviderOptions.VertexAI:
            self.LLM_provider = OpenAI(
                temperature=temperature,
                max_tokens=max_output_tokens,
                client=OpenAIclient,
                model=llm_model
                )
        else:
            raise KeyError(
                f'''LLM_PROVIDER environment variable not found!
                Needs to be one of the following values:
                {ProviderOptions.list_enums()}'''
                )

        # Set Embedding Provider
        if ENV_llm_provider == ProviderOptions.BasedLibrarianServer:
            raise NotImplementedError
        if ENV_llm_provider == ProviderOptions.LlamaCPP:
            raise NotImplementedError
        if ENV_llm_provider == ProviderOptions.VertexAI:
            self.Embeddings_provider = VertexAIEmbeddings()
        if ENV_llm_provider == ProviderOptions.VertexAI:
            if embeddings_model == "NA":
                self.Embeddings_provider = OpenAIEmbeddings(
                    client=OpenAIclient,
                    model="text-embedding-ada-002"
                    )
            elif embeddings_model != "NA":
                self.Embeddings_provider = OpenAIEmbeddings(
                    client=OpenAIclient,
                    model=embeddings_model
                    )
