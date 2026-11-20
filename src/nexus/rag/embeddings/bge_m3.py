import logging

from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class BGEM3Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 12,
        max_length: int = 8192,
    ):
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            devices=[device],
        )

        self.batch_size = batch_size
        self.max_length = max_length

        logger.info("Loaded embedding model: %s", model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        if not texts:
            return []

        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return output["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:

        output = self.model.encode(
            [text],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return output["dense_vecs"][0].tolist()


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    embedding_model = BGEM3Embeddings(
        model_name=os.getenv("", "BAAI/bge-m3"),
        device="cpu",
        use_fp16=False,
    )

    # Test embedding a single query
    query_embedding = embedding_model.embed_query("Hello, world!")
    print("Query embedding:", len(query_embedding))

    # Test embedding multiple documents
    documents = ["Document 1", "Document 2", "Document 3"]
    document_embeddings = embedding_model.embed_documents(documents)
    print("Document embeddings:", len(document_embeddings[0]))
