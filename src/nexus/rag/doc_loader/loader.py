from pathlib import Path

from langchain_core.documents import Document

from nexus.rag.doc_loader.docling_provider import DoclingDocumentLoader


class DocumentLoader:
    def __init__(self, provider: str = "docling"):
        self.providers = {
            "docling": DoclingDocumentLoader,
        }

        if provider not in self.providers:
            raise ValueError(f"Unsupported document loader: {provider}")

        self.loader = self.providers[provider]()

    def load(self, file_path: str | Path) -> list[Document]:
        return self.loader.load(file_path)


if __name__ == "__main__":
    from nexus.rag.doc_loader.loader import DocumentLoader

    loader = DocumentLoader("docling")
    docs = loader.load(r"D:\IAOSC\projects\Codes\Nexus\data\test_data\Cloud-Basic.pdf")

    print(f"Number of documents: {len(docs)}")

    for doc in docs:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)
