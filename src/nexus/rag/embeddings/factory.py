from typing import Literal

from nexus.rag.embeddings.bge_m3 import BGEM3Embeddings
from nexus.rag.embeddings.gemini import GeminiEmbeddings

model = Literal["gemini", "bge_m3"]


def get_embeddings(provider: model = "bge_m3") -> GeminiEmbeddings | BGEM3Embeddings:

    if provider == "gemini":
        return GeminiEmbeddings()

    elif provider == "bge_m3":
        return BGEM3Embeddings(
            device="cpu",
            use_fp16=False,
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
