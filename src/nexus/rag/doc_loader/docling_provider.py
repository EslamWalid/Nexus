
from pathlib import Path

from langchain_docling.loader import DoclingLoader, ExportType
from langchain_core.documents import Document



class DoclingDocumentLoader:

    def __init__(self):
        self.export_type = ExportType.MARKDOWN

    def load(self, file_path: str | Path)-> list[Document]:

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        loader = DoclingLoader(
            file_path=str(path),
            export_type=self.export_type,
        )

        documents = loader.load()

        # Add source information to metadata
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
        
