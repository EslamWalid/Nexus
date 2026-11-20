from nexus.rag.embeddings.bge_m3 import BGEM3Embeddings
from nexus.rag.embeddings.gemini import GeminiEmbeddings


def get_embeddings(provider: str):

    if provider == "gemini":
        return GeminiEmbeddings()

    elif provider == "bge_m3":
        return BGEM3Embeddings(
            device="cpu",
            use_fp16=False,
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
