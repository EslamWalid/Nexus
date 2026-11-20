# tests/rag/embeddings/test_bge_m3.py

import numpy as np

from nexus.rag.embeddings import bge_m3


class FakeBGEM3Model:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.encoded_texts = None

    def encode(self, texts, **kwargs):
        self.encoded_texts = texts

        return {"dense_vecs": np.ones((len(texts), 1024))}


def test_model_initialization(monkeypatch):
    monkeypatch.setattr(
        bge_m3,
        "BGEM3FlagModel",
        FakeBGEM3Model,
    )

    embeddings = bge_m3.BGEM3Embeddings(
        device="cpu",
        use_fp16=False,
    )

    assert isinstance(embeddings.model, FakeBGEM3Model)
    assert embeddings.model.kwargs["use_fp16"] is False
    assert embeddings.model.kwargs["devices"] == ["cpu"]


def test_embed_query(monkeypatch):
    monkeypatch.setattr(
        bge_m3,
        "BGEM3FlagModel",
        FakeBGEM3Model,
    )

    embeddings = bge_m3.BGEM3Embeddings()

    result = embeddings.embed_query("Hello world")

    assert isinstance(result, list)
    assert len(result) == 1024
    assert embeddings.model.encoded_texts == ["Hello world"]


def test_embed_documents(monkeypatch):
    monkeypatch.setattr(
        bge_m3,
        "BGEM3FlagModel",
        FakeBGEM3Model,
    )

    embeddings = bge_m3.BGEM3Embeddings()

    documents = ["Document 1", "Document 2", "Document 3"]

    result = embeddings.embed_documents(documents)

    assert len(result) == 3
    assert all(len(vector) == 1024 for vector in result)


def test_embed_documents_empty_input(monkeypatch):
    monkeypatch.setattr(
        bge_m3,
        "BGEM3FlagModel",
        FakeBGEM3Model,
    )

    embeddings = bge_m3.BGEM3Embeddings()

    result = embeddings.embed_documents([])

    assert result == []
    assert embeddings.model.encoded_texts is None
