from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from nexus.rag.text_splitter.recursive_splitter import RecursiveChunking
from nexus.rag.text_splitter.semantic import SemanticChunking

SplitterType = Literal["semantic", "recursive"]


class DocumentSplitter:
    def __init__(
        self,
        splitter_type: SplitterType = "recursive",
        embeddings: Embeddings | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ):

        if splitter_type == "semantic":
            if embeddings is None:
                raise ValueError("Embeddings are required for semantic chunking.")

            self.splitter = SemanticChunking(
                embeddings=embeddings,
                **kwargs,
            )

        elif splitter_type == "recursive":
            self.splitter = RecursiveChunking(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported splitter type: {splitter_type}")

    def split_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Document]:

        return self.splitter.split_text(text, metadata)

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:

        return self.splitter.split_documents(documents)


if __name__ == "__main__":
    from nexus.rag.doc_loader.loader import DocumentLoader
    from nexus.rag.embeddings.factory import get_embeddings

    embeddings = get_embeddings("bge_m3")
    chunker = DocumentSplitter()
    loader = DocumentLoader()
    docs = loader.load(r"D:\IAOSC\projects\Codes\Nexus\data\test_data\Cloud-Basic.pdf")
    for doc in docs:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)
    documents = chunker.split_text(doc.page_content)
    print(f"Number of chunks: {len(documents)}")
    print("Content of first chunk:", documents)
