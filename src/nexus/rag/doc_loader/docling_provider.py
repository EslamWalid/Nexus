from pathlib import Path
from threading import Lock

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader, ExportType


class DoclingDocumentLoader:
    def __init__(self):
        self.converter = DocumentConverter()
        self.export_type = ExportType.MARKDOWN
        self.lock = Lock()

    def load(self, file_path: str | Path) -> list[Document]:

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file: {path}")

        loader = DoclingLoader(
            file_path=str(path),
            converter=self.converter,
            export_type=self.export_type,
        )

        # Serialize conversion requests using this converter
        with self.lock:
            documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = str(path)

        return documents


if __name__ == "__main__":
    loader = DoclingDocumentLoader()

    docs = loader.load(r"D:\IAOSC\projects\Codes\Nexus\data\test_data\Cloud-Basic.pdf")

    print(f"Number of documents: {len(docs)}")

    for doc in docs:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)
