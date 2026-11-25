from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from nexus.rag.text_splitter.semantic import SemanticChunking


@pytest.fixture
def embeddings():
    """Provide fake embeddings without external API calls."""
    return FakeEmbeddings(size=768)


@pytest.fixture
def semantic_chunker(embeddings):
    """Create a SemanticChunking instance."""
    return SemanticChunking(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        buffer_size=1,
    )


@pytest.fixture
def sample_text():
    return (
        "Artificial intelligence is transforming the world. "
        "Machine learning is a subset of artificial intelligence. "
        "Deep learning uses neural networks to learn complex patterns."
    )


def test_initialization(semantic_chunker):
    """Test that the semantic chunker initializes correctly."""
    assert semantic_chunker.chunker is not None


def test_initialization_with_custom_parameters(embeddings):
    """Test initialization with custom configuration."""
    chunker = SemanticChunking(
        embeddings=embeddings,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=2.0,
        buffer_size=2,
    )

    assert chunker.chunker is not None


def test_split_text_returns_documents(semantic_chunker, sample_text):
    """Test that split_text returns a list of Documents."""
    result = semantic_chunker.split_text(sample_text)

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(doc, Document) for doc in result)


def test_split_text_preserves_metadata(semantic_chunker, sample_text):
    """Test that metadata is preserved in generated chunks."""
    metadata = {
        "source": "test.pdf",
        "page": 1,
    }

    result = semantic_chunker.split_text(
        text=sample_text,
        metadata=metadata,
    )

    assert len(result) > 0

    for doc in result:
        assert doc.metadata == metadata


def test_split_text_without_metadata(semantic_chunker, sample_text):
    """Test splitting text without metadata."""
    result = semantic_chunker.split_text(sample_text)

    assert len(result) > 0

    for doc in result:
        assert isinstance(doc.metadata, dict)


def test_split_documents_returns_documents(semantic_chunker):
    """Test splitting a list of documents."""
    documents = [
        Document(
            page_content=(
                "Artificial intelligence is transforming technology. "
                "Machine learning helps computers learn from data."
            ),
            metadata={"source": "doc1.pdf"},
        ),
        Document(
            page_content=(
                "Deep learning uses neural networks. "
                "Neural networks consist of interconnected layers."
            ),
            metadata={"source": "doc2.pdf"},
        ),
    ]

    result = semantic_chunker.split_documents(documents)

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(doc, Document) for doc in result)


def test_split_documents_preserves_metadata(semantic_chunker):
    """Test that document metadata is preserved after splitting."""
    documents = [
        Document(
            page_content=(
                "Python is a programming language. "
                "It is widely used in artificial intelligence."
            ),
            metadata={"source": "python.pdf", "page": 5},
        )
    ]

    result = semantic_chunker.split_documents(documents)

    assert len(result) > 0

    for doc in result:
        assert doc.metadata["source"] == "python.pdf"
        assert doc.metadata["page"] == 5


def test_split_documents_empty_list(semantic_chunker):
    """Test splitting an empty document list."""
    result = semantic_chunker.split_documents([])

    assert result == []


def test_split_text_calls_create_documents(
    semantic_chunker,
    sample_text,
):
    """Test that split_text delegates to create_documents."""
    mock_result = [Document(page_content="Mock chunk", metadata={"source": "test"})]

    semantic_chunker.chunker.create_documents = MagicMock(return_value=mock_result)

    metadata = {"source": "test"}

    result = semantic_chunker.split_text(
        sample_text,
        metadata=metadata,
    )

    semantic_chunker.chunker.create_documents.assert_called_once_with(
        texts=[sample_text],
        metadatas=[metadata],
    )

    assert result == mock_result


def test_split_text_with_empty_metadata(semantic_chunker, sample_text):
    """Test that empty metadata is passed as None."""
    semantic_chunker.chunker.create_documents = MagicMock(return_value=[])

    semantic_chunker.split_text(sample_text, metadata={})

    semantic_chunker.chunker.create_documents.assert_called_once_with(
        texts=[sample_text],
        metadatas=None,
    )
