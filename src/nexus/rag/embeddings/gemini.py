import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from nexus.config import EMBEDDING_MODEL

ROOT_DIR = Path.cwd()

load_dotenv(ROOT_DIR / ".env", override=True)

logger = logging.getLogger(__name__)


class GeminiEmbeddings(Embeddings):
    def __init__(self):
        model = EMBEDDING_MODEL
        api_keys = [
            os.getenv("EMBEDDING_API_KEY1"),
            os.getenv("EMBEDDING_API_KEY2"),
        ]
        # Remove missing API keys
        api_keys = [key for key in api_keys if key]

        if not api_keys:
            raise ValueError("No embedding API keys configured")

        self.clients = [
            GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=key,
            )
            for key in api_keys
        ]

    def _execute(self, method, data):
        last_error = None

        for index, client in enumerate(self.clients):
            try:
                func = getattr(client, method)

                if method == "embed_query":
                    return func(data)

                return func(data)

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Gemini key %s failed: %s",
                    index + 1,
                    exc,
                )

        raise RuntimeError("All Gemini API keys failed") from last_error

    def embed_query(self, text: str) -> list[float]:
        return self._execute("embed_query", text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._execute("embed_documents", texts)


if __name__ == "__main__":
    embeddings = GeminiEmbeddings()
    vector = embeddings.embed_query("hello, world!")
    print(len(vector))
