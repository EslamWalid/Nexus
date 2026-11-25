from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunking:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Sequence[str] | None = None,
        keep_separator: bool | str = True,
    ):
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
            or [
                "\n\n",
                "\n",
                ". ",
                "؟ ",
                "! ",
                " ",
                "",
            ],
            keep_separator=keep_separator,
            length_function=len,
        )

    def split_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Document]:

        metadatas = [metadata] if metadata is not None else None

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
    splitter = RecursiveChunking(chunk_size=100, chunk_overlap=20)

    text = (
        """This is a sample text that will be split into smaller chunks. """
        """The RecursiveCharacterTextSplitter will handle 
            the splitting based on the specified chunk size and overlap."""
        """This allows for better processing of large texts in 
            applications such as natural language processing 
            and information retrieval."""
    )

    chunks = splitter.split_text(text)
    print("Number of chunks:", len(chunks))
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}")
