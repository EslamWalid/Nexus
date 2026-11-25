import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_docling.loader import ExportType

from nexus.rag.doc_loader import docling_provider as module

DoclingDocumentLoader = module.DoclingDocumentLoader


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_converter_cls(monkeypatch):
    """Mock لـ DocumentConverter."""
    mock_cls = MagicMock(name="DocumentConverter")
    monkeypatch.setattr(module, "DocumentConverter", mock_cls)
    return mock_cls


@pytest.fixture
def mock_loader_cls(monkeypatch):

    mock_cls = MagicMock(name="DoclingLoader")
    mock_cls.return_value.load.return_value = [
        Document(page_content="Hello", metadata={"page": 1}),
        Document(page_content="World", metadata={"page": 2}),
    ]
    monkeypatch.setattr(module, "DoclingLoader", mock_cls)
    return mock_cls


@pytest.fixture
def loader(mock_converter_cls, mock_loader_cls):
    return DoclingDocumentLoader()


@pytest.fixture
def sample_file(tmp_path) -> Path:
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"%PDF-1.4 fake content")
    return f


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #
class TestInit:
    def test_creates_converter(self, mock_converter_cls, mock_loader_cls):
        loader = DoclingDocumentLoader()

        mock_converter_cls.assert_called_once_with()
        assert loader.converter is mock_converter_cls.return_value

    def test_export_type_is_markdown(self, loader):
        assert loader.export_type == ExportType.MARKDOWN

    def test_has_lock(self, loader):
        assert hasattr(loader.lock, "acquire")
        assert hasattr(loader.lock, "release")
        assert not loader.lock.locked()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_file_not_found(self, loader, tmp_path):
        missing = tmp_path / "missing.pdf"

        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load(missing)

    def test_file_not_found_message_contains_path(self, loader, tmp_path):
        missing = tmp_path / "missing.pdf"

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load(missing)

        assert str(missing.resolve()) in str(exc_info.value)

    def test_directory_raises_value_error(self, loader, tmp_path):
        with pytest.raises(ValueError, match="Expected a file"):
            loader.load(tmp_path)

    def test_docling_loader_not_created_when_file_missing(
        self, loader, mock_loader_cls, tmp_path
    ):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nope.pdf")

        mock_loader_cls.assert_not_called()

    def test_docling_loader_not_created_when_path_is_directory(
        self, loader, mock_loader_cls, tmp_path
    ):
        with pytest.raises(ValueError):
            loader.load(tmp_path)

        mock_loader_cls.assert_not_called()


# --------------------------------------------------------------------------- #
# load() - happy path
# --------------------------------------------------------------------------- #
class TestLoad:
    def test_returns_documents(self, loader, sample_file):
        docs = loader.load(sample_file)

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert [d.page_content for d in docs] == ["Hello", "World"]

    def test_accepts_str_path(self, loader, sample_file):
        docs = loader.load(str(sample_file))

        assert len(docs) == 2

    def test_accepts_path_object(self, loader, sample_file):
        docs = loader.load(sample_file)

        assert len(docs) == 2

    def test_docling_loader_called_with_correct_args(
        self, loader, mock_loader_cls, sample_file
    ):
        loader.load(sample_file)

        mock_loader_cls.assert_called_once_with(
            file_path=str(sample_file.resolve()),
            converter=loader.converter,
            export_type=ExportType.MARKDOWN,
        )

    def test_loader_load_called_once(self, loader, mock_loader_cls, sample_file):
        loader.load(sample_file)

        mock_loader_cls.return_value.load.assert_called_once_with()

    def test_returns_empty_list_when_no_documents(
        self, loader, mock_loader_cls, sample_file
    ):
        mock_loader_cls.return_value.load.return_value = []

        assert loader.load(sample_file) == []


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
class TestMetadata:
    def test_source_metadata_is_set_on_all_docs(self, loader, sample_file):
        docs = loader.load(sample_file)

        for doc in docs:
            assert doc.metadata["source"] == str(sample_file.resolve())

    def test_existing_metadata_is_preserved(self, loader, sample_file):
        docs = loader.load(sample_file)

        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

    def test_existing_source_is_overridden(self, loader, mock_loader_cls, sample_file):
        mock_loader_cls.return_value.load.return_value = [
            Document(page_content="x", metadata={"source": "old/path.pdf"})
        ]

        docs = loader.load(sample_file)

        assert docs[0].metadata["source"] == str(sample_file.resolve())


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
class TestPathResolution:
    def test_relative_path_is_resolved(
        self, loader, mock_loader_cls, sample_file, monkeypatch
    ):
        monkeypatch.chdir(sample_file.parent)

        docs = loader.load(sample_file.name)

        expected = str(sample_file.resolve())
        assert mock_loader_cls.call_args.kwargs["file_path"] == expected
        assert docs[0].metadata["source"] == expected

    def test_tilde_is_expanded(self, loader, mock_loader_cls, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        f = tmp_path / "home_file.pdf"
        f.write_bytes(b"data")

        docs = loader.load("~/home_file.pdf")

        assert docs[0].metadata["source"] == str(f.resolve())
        assert "~" not in mock_loader_cls.call_args.kwargs["file_path"]

    def test_dotdot_is_resolved(self, loader, mock_loader_cls, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        f = tmp_path / "file.pdf"
        f.write_bytes(b"data")

        loader.load(sub / ".." / "file.pdf")

        assert mock_loader_cls.call_args.kwargs["file_path"] == str(f.resolve())


# --------------------------------------------------------------------------- #
# Locking / thread safety
# --------------------------------------------------------------------------- #
class TestLocking:
    def test_lock_is_held_during_conversion(self, loader, mock_loader_cls, sample_file):
        observed = {}

        def fake_load():
            observed["locked"] = loader.lock.locked()
            return [Document(page_content="x")]

        mock_loader_cls.return_value.load.side_effect = fake_load

        loader.load(sample_file)

        assert observed["locked"] is True

    def test_lock_is_released_after_load(self, loader, sample_file):
        loader.load(sample_file)

        assert not loader.lock.locked()

    def test_lock_is_released_if_loader_raises(
        self, loader, mock_loader_cls, sample_file
    ):
        mock_loader_cls.return_value.load.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            loader.load(sample_file)

        assert not loader.lock.locked()

    def test_loader_exception_is_propagated(self, loader, mock_loader_cls, sample_file):
        mock_loader_cls.return_value.load.side_effect = ValueError("bad pdf")

        with pytest.raises(ValueError, match="bad pdf"):
            loader.load(sample_file)

    def test_concurrent_loads_are_serialized(
        self, loader, mock_loader_cls, sample_file
    ):
        state = {"active": 0, "max_active": 0}
        state_guard = threading.Lock()

        def slow_load():
            with state_guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with state_guard:
                state["active"] -= 1
            return [Document(page_content="x")]

        mock_loader_cls.return_value.load.side_effect = slow_load

        results = []
        errors = []

        def worker():
            try:
                results.append(loader.load(sample_file))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 5
        assert state["max_active"] == 1
