from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import (
    SemanticChunker as LCSemanticChunker,
)

BreakpointType = Literal[
    "percentile",
    "standard_deviation",
    "interquartile",
    "gradient",
]


class SemanticChunking:
    def __init__(
        self,
        embeddings: Embeddings,
        breakpoint_threshold_type: BreakpointType = "percentile",
        breakpoint_threshold_amount: float | None = None,
        buffer_size: int = 1,
    ):
        self.chunker = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
        )

    def split_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Document]:
        metadatas = [metadata] if metadata else None

        return self.chunker.create_documents(
            texts=[text],
            metadatas=metadatas,
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        return self.chunker.split_documents(documents)


if __name__ == "__main__":
    from nexus.rag.doc_loader.loader import DocumentLoader
    from nexus.rag.embeddings.factory import get_embeddings

    embeddings = get_embeddings("bge_m3")
    chunker = SemanticChunking(embeddings=embeddings)
    loader = DocumentLoader()
    docs = loader.load(r"D:\IAOSC\projects\Codes\Nexus\data\test_data\Cloud-Basic.pdf")
    for doc in docs:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)
    documents = chunker.split_text(doc.page_content)
    print(f"Number of chunks: {len(documents)}")
    print("Content of first chunk:", documents)
