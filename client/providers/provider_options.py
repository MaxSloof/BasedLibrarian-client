from enum import Enum


class ProviderOptions(Enum):
    BasedLibrarianServer = "blserver"
    VertexAI = "vertexai"
    OpenAI = "openai"
    LlamaCPP = "llamacpp"

    @classmethod
    def list_enums(cls):
        return [enum.value for enum in cls]
