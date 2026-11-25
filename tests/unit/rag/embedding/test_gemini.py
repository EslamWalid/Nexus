# tests/rag/embeddings/test_gemini.py

import pytest

from nexus.rag.embeddings import gemini


class FakeGeminiClient:
    def __init__(self, model, google_api_key):
        self.api_key = google_api_key

    def embed_query(self, text):
        if self.api_key == "key1":
            raise Exception("API key failed")

        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        if self.api_key == "key1":
            raise Exception("API key failed")

        return [[0.1, 0.2, 0.3] for _ in texts]


def test_no_api_keys(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY1", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY2", raising=False)

    with pytest.raises(ValueError, match="No embedding API keys"):
        gemini.GeminiEmbeddings()


def test_clients_initialization(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY1", "key1")
    monkeypatch.setenv("EMBEDDING_API_KEY2", "key2")

    monkeypatch.setattr(
        gemini,
        "GoogleGenerativeAIEmbeddings",
        FakeGeminiClient,
    )

    embeddings = gemini.GeminiEmbeddings()

    assert len(embeddings.clients) == 2


def test_embed_query_fallback(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY1", "key1")
    monkeypatch.setenv("EMBEDDING_API_KEY2", "key2")

    monkeypatch.setattr(
        gemini,
        "GoogleGenerativeAIEmbeddings",
        FakeGeminiClient,
    )

    embeddings = gemini.GeminiEmbeddings()

    result = embeddings.embed_query("Hello")

    assert result == [0.1, 0.2, 0.3]


def test_embed_documents_fallback(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY1", "key1")
    monkeypatch.setenv("EMBEDDING_API_KEY2", "key2")

    monkeypatch.setattr(
        gemini,
        "GoogleGenerativeAIEmbeddings",
        FakeGeminiClient,
    )

    embeddings = gemini.GeminiEmbeddings()

    result = embeddings.embed_documents(["Doc 1", "Doc 2"])

    assert len(result) == 2


def test_all_api_keys_fail(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY1", "key1")

    class FailingClient:
        def __init__(self, **kwargs):
            pass

        def embed_query(self, text):
            raise Exception("API failure")

    monkeypatch.setattr(
        gemini,
        "GoogleGenerativeAIEmbeddings",
        FailingClient,
    )

    embeddings = gemini.GeminiEmbeddings()

    with pytest.raises(RuntimeError, match="All Gemini API keys failed"):
        embeddings.embed_query("Hello")
