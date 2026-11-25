from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from nexus.rag.doc_loader import loader as loader_module
from nexus.rag.doc_loader.loader import DocumentLoader


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_docling_cls(monkeypatch):
    """Replace DoclingDocumentLoader inside loader.py with a mock class."""
    mock_cls = MagicMock(name="DoclingDocumentLoader")
    mock_cls.return_value.load.return_value = [
        Document(page_content="Hello", metadata={"source": "a.pdf"}),
        Document(page_content="World", metadata={"source": "a.pdf"}),
    ]
    monkeypatch.setattr(loader_module, "DoclingDocumentLoader", mock_cls)
    return mock_cls


@pytest.fixture
def loader(mock_docling_cls):
    return DocumentLoader()


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #
class TestInit:
    def test_default_provider_is_docling(self, mock_docling_cls):
        loader = DocumentLoader()

        mock_docling_cls.assert_called_once_with()
        assert loader.loader is mock_docling_cls.return_value

    def test_explicit_docling_provider(self, mock_docling_cls):
        loader = DocumentLoader("docling")

        mock_docling_cls.assert_called_once_with()
        assert loader.loader is mock_docling_cls.return_value

    def test_provider_passed_as_keyword(self, mock_docling_cls):
        loader = DocumentLoader(provider="docling")

        assert loader.loader is mock_docling_cls.return_value

    def test_providers_registry_contains_docling(self, loader, mock_docling_cls):
        assert "docling" in loader.providers
        assert loader.providers["docling"] is mock_docling_cls

    def test_provider_is_instantiated_only_once(self, mock_docling_cls):
        loader = DocumentLoader()

        # Loading multiple times must reuse the same provider instance.
        loader.load("a.pdf")
        loader.load("b.pdf")

        mock_docling_cls.assert_called_once_with()


# --------------------------------------------------------------------------- #
# Unsupported providers
# --------------------------------------------------------------------------- #
class TestUnsupportedProvider:
    def test_unknown_provider_raises_value_error(self, mock_docling_cls):
        with pytest.raises(ValueError, match="Unsupported document loader"):
            DocumentLoader("unknown")

    def test_error_message_contains_provider_name(self, mock_docling_cls):
        with pytest.raises(ValueError) as exc_info:
            DocumentLoader("my_custom_provider")

        assert "my_custom_provider" in str(exc_info.value)

    @pytest.mark.parametrize(
        "provider",
        ["", "pdf", "Docling", "DOCLING", " docling", "docling ", None],
    )
    def test_invalid_providers_are_rejected(self, mock_docling_cls, provider):
        # Provider names are case-sensitive and must match exactly.
        with pytest.raises(ValueError, match="Unsupported document loader"):
            DocumentLoader(provider)

    def test_no_provider_is_instantiated_on_invalid_name(self, mock_docling_cls):
        with pytest.raises(ValueError):
            DocumentLoader("unknown")

        mock_docling_cls.assert_not_called()


# --------------------------------------------------------------------------- #
# load()
# --------------------------------------------------------------------------- #
class TestLoad:
    def test_delegates_to_provider_with_str_path(self, loader, mock_docling_cls):
        loader.load("some/file.pdf")

        mock_docling_cls.return_value.load.assert_called_once_with("some/file.pdf")

    def test_delegates_to_provider_with_path_object(self, loader, mock_docling_cls):
        path = Path("some/file.pdf")

        loader.load(path)

        mock_docling_cls.return_value.load.assert_called_once_with(path)

    def test_path_is_passed_through_unchanged(self, loader, mock_docling_cls):
        path = Path("~/docs/../docs/file.pdf")

        loader.load(path)

        # The facade must not normalize or modify the path itself.
        passed = mock_docling_cls.return_value.load.call_args.args[0]
        assert passed is path

    def test_returns_provider_result(self, loader, mock_docling_cls):
        expected = mock_docling_cls.return_value.load.return_value

        result = loader.load("file.pdf")

        assert result is expected
        assert [d.page_content for d in result] == ["Hello", "World"]

    def test_returns_empty_list_when_provider_returns_nothing(
        self, loader, mock_docling_cls
    ):
        mock_docling_cls.return_value.load.return_value = []

        assert loader.load("empty.pdf") == []

    def test_each_call_is_delegated(self, loader, mock_docling_cls):
        loader.load("a.pdf")
        loader.load("b.pdf")

        calls = mock_docling_cls.return_value.load.call_args_list
        assert [c.args[0] for c in calls] == ["a.pdf", "b.pdf"]


# --------------------------------------------------------------------------- #
# Error propagation
# --------------------------------------------------------------------------- #
class TestErrorPropagation:
    def test_file_not_found_is_propagated(self, loader, mock_docling_cls):
        mock_docling_cls.return_value.load.side_effect = FileNotFoundError(
            "File not found: missing.pdf"
        )

        with pytest.raises(FileNotFoundError, match="missing.pdf"):
            loader.load("missing.pdf")

    def test_value_error_is_propagated(self, loader, mock_docling_cls):
        mock_docling_cls.return_value.load.side_effect = ValueError("Expected a file")

        with pytest.raises(ValueError, match="Expected a file"):
            loader.load("some_dir")

    def test_unexpected_exception_is_propagated(self, loader, mock_docling_cls):
        mock_docling_cls.return_value.load.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            loader.load("file.pdf")


# --------------------------------------------------------------------------- #
# Integration with the real provider class (no Docling models loaded)
# --------------------------------------------------------------------------- #
class TestWithRealProviderClass:
    def test_registry_points_to_real_docling_class(self):
        """Sanity check that the unpatched registry references the real class."""
        from nexus.rag.doc_loader.docling_provider import DoclingDocumentLoader

        # Avoid instantiating the real provider (it creates a DocumentConverter),
        # so only inspect the class used by the module.
        assert loader_module.DoclingDocumentLoader is DoclingDocumentLoader
