# tests/rag/embeddings/test_factory.py

import pytest

from nexus.rag.embeddings import factory


def test_get_gemini_embeddings(monkeypatch):
    class FakeGemini:
        pass

    monkeypatch.setattr(
        factory,
        "GeminiEmbeddings",
        FakeGemini,
    )

    result = factory.get_embeddings("gemini")

    assert isinstance(result, FakeGemini)


def test_get_bge_m3_embeddings(monkeypatch):
    class FakeBGE:
        def __init__(self, device, use_fp16):
            self.device = device
            self.use_fp16 = use_fp16

    monkeypatch.setattr(
        factory,
        "BGEM3Embeddings",
        FakeBGE,
    )

    result = factory.get_embeddings("bge_m3")

    assert isinstance(result, FakeBGE)
    assert result.device == "cpu"
    assert result.use_fp16 is False


def test_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        factory.get_embeddings("unknown")
