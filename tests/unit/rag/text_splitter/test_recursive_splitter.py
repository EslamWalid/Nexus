import pytest
from langchain_core.documents import Document

from nexus.rag.text_splitter.splitter import RecursiveChunking


@pytest.fixture
def splitter():
    return RecursiveChunking(
        chunk_size=100,
        chunk_overlap=20,
    )


class TestRecursiveChunking:
    def test_initialization(self, splitter):
        assert splitter.chunker is not None
        assert splitter.chunker._chunk_size == 100
        assert splitter.chunker._chunk_overlap == 20

    def test_split_text_returns_documents(self, splitter):
        text = "This is a sample text for testing the recursive splitter."

        chunks = splitter.split_text(text)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_split_text_creates_multiple_chunks(self):
        splitter = RecursiveChunking(
            chunk_size=30,
            chunk_overlap=5,
        )

        text = "This is a long text that should be split into multiple chunks."

        chunks = splitter.split_text(text)

        assert len(chunks) > 1

    def test_chunk_size_limit(self):
        splitter = RecursiveChunking(
            chunk_size=50,
            chunk_overlap=10,
        )

        text = "This is a long text. " * 20

        chunks = splitter.split_text(text)

        assert all(len(chunk.page_content) <= 50 for chunk in chunks)

    def test_metadata_preservation(self, splitter):
        metadata = {
            "source": "test.pdf",
            "page": 1,
            "category": "AI",
        }

        text = "This is a sample text. " * 20

        chunks = splitter.split_text(
            text=text,
            metadata=metadata,
        )

        assert len(chunks) > 1

        for chunk in chunks:
            assert chunk.metadata == metadata

    def test_split_text_without_metadata(self, splitter):
        text = "This is a sample text. " * 10

        chunks = splitter.split_text(text)

        assert all(chunk.metadata == {} for chunk in chunks)

    def test_split_documents(self, splitter):
        documents = [
            Document(
                page_content="First document content. " * 10,
                metadata={"source": "first.pdf", "page": 1},
            ),
            Document(
                page_content="Second document content. " * 10,
                metadata={"source": "second.pdf", "page": 2},
            ),
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) > len(documents)

        assert all(isinstance(chunk, Document) for chunk in chunks)

        assert all(
            chunk.metadata["source"] in ["first.pdf", "second.pdf"] for chunk in chunks
        )

    def test_split_documents_preserves_metadata(self, splitter):
        documents = [
            Document(
                page_content="First document content. " * 10,
                metadata={"source": "first.pdf", "page": 1},
            ),
            Document(
                page_content="Second document content. " * 10,
                metadata={"source": "second.pdf", "page": 2},
            ),
        ]

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            if chunk.metadata["source"] == "first.pdf":
                assert chunk.metadata["page"] == 1

            elif chunk.metadata["source"] == "second.pdf":
                assert chunk.metadata["page"] == 2

    def test_empty_text(self, splitter):
        chunks = splitter.split_text("")

        assert chunks == []

    def test_short_text(self, splitter):
        text = "Short text."

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].page_content == text

    def test_custom_separators(self):
        splitter = RecursiveChunking(
            chunk_size=30,
            chunk_overlap=0,
            separators=["|", " ", ""],
        )

        text = "First section|Second section|Third section"

        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 30 for chunk in chunks)

    def test_original_documents_are_not_modified(self, splitter):
        original = Document(
            page_content="Some content. " * 10,
            metadata={"source": "test.pdf"},
        )

        original_content = original.page_content
        original_metadata = original.metadata.copy()

        splitter.split_documents([original])

        assert original.page_content == original_content
        assert original.metadata == original_metadata
