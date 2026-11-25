from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from nexus.rag.text_splitter.splitter import DocumentSplitter


class TestDocumentSplitter:
    # -------------------------
    # Initialization
    # -------------------------

    @patch("nexus.rag.text_splitter.splitter.RecursiveChunking")
    def test_recursive_initialization(self, mock_recursive):

        splitter = DocumentSplitter(
            splitter_type="recursive",
            chunk_size=500,
            chunk_overlap=100,
        )

        mock_recursive.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=100,
        )

        assert splitter.splitter is mock_recursive.return_value

    @patch("nexus.rag.text_splitter.splitter.SemanticChunking")
    def test_semantic_initialization(self, mock_semantic):

        embeddings = MagicMock()

        splitter = DocumentSplitter(
            splitter_type="semantic",
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            buffer_size=2,
        )

        mock_semantic.assert_called_once_with(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            buffer_size=2,
        )

        assert splitter.splitter is mock_semantic.return_value

    def test_recursive_is_default_splitter(self):

        with patch(
            "nexus.rag.text_splitter.splitter.RecursiveChunking"
        ) as mock_recursive:
            splitter = DocumentSplitter()

            mock_recursive.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
            )

            assert splitter.splitter is mock_recursive.return_value

    def test_semantic_without_embeddings_raises_error(self):

        with pytest.raises(
            ValueError,
            match="Embeddings are required for semantic chunking",
        ):
            DocumentSplitter(splitter_type="semantic")

    def test_invalid_splitter_type_raises_error(self):

        with pytest.raises(
            ValueError,
            match="Unsupported splitter type",
        ):
            DocumentSplitter(splitter_type="invalid")

    # -------------------------
    # split_text
    # -------------------------

    def test_split_text_delegates_to_recursive_splitter(self):

        with patch(
            "nexus.rag.text_splitter.splitter.RecursiveChunking"
        ) as mock_recursive:
            expected_chunks = [
                Document(page_content="Chunk 1"),
                Document(page_content="Chunk 2"),
            ]

            mock_recursive.return_value.split_text.return_value = expected_chunks

            splitter = DocumentSplitter()

            metadata = {"source": "test.pdf"}

            result = splitter.split_text(
                text="Sample text",
                metadata=metadata,
            )

            mock_recursive.return_value.split_text.assert_called_once_with(
                "Sample text",
                metadata,
            )

            assert result == expected_chunks

    def test_split_text_delegates_to_semantic_splitter(self):

        with patch(
            "nexus.rag.text_splitter.splitter.SemanticChunking"
        ) as mock_semantic:
            embeddings = MagicMock()

            expected_chunks = [
                Document(page_content="Semantic chunk"),
            ]

            mock_semantic.return_value.split_text.return_value = expected_chunks

            splitter = DocumentSplitter(
                splitter_type="semantic",
                embeddings=embeddings,
            )

            metadata = {"source": "test.pdf"}

            result = splitter.split_text(
                "Sample text",
                metadata,
            )

            mock_semantic.return_value.split_text.assert_called_once_with(
                "Sample text",
                metadata,
            )

            assert result == expected_chunks

    # -------------------------
    # split_documents
    # -------------------------

    def test_split_documents_delegates_to_recursive_splitter(self):

        with patch(
            "nexus.rag.text_splitter.splitter.RecursiveChunking"
        ) as mock_recursive:
            documents = [
                Document(
                    page_content="First document",
                    metadata={"source": "first.pdf"},
                ),
                Document(
                    page_content="Second document",
                    metadata={"source": "second.pdf"},
                ),
            ]

            expected_chunks = [
                Document(page_content="Chunk 1"),
            ]

            mock_recursive.return_value.split_documents.return_value = expected_chunks

            splitter = DocumentSplitter()

            result = splitter.split_documents(documents)

            mock_recursive.return_value.split_documents.assert_called_once_with(
                documents
            )

            assert result == expected_chunks

    def test_split_documents_delegates_to_semantic_splitter(self):

        with patch(
            "nexus.rag.text_splitter.splitter.SemanticChunking"
        ) as mock_semantic:
            embeddings = MagicMock()

            documents = [
                Document(
                    page_content="Sample document",
                    metadata={"source": "test.pdf"},
                )
            ]

            expected_chunks = [
                Document(page_content="Semantic chunk"),
            ]

            mock_semantic.return_value.split_documents.return_value = expected_chunks

            splitter = DocumentSplitter(
                splitter_type="semantic",
                embeddings=embeddings,
            )

            result = splitter.split_documents(documents)

            mock_semantic.return_value.split_documents.assert_called_once_with(
                documents
            )

            assert result == expected_chunks
