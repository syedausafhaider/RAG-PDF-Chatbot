import os
from typing import List, Union
import numpy as np

# FAISS imports
import faiss

# Chroma imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class Retriever:
    def __init__(self, vector_store_type: str, faiss_index_path: str = None, chroma_dir: str = None):
        """
        Initialize the retriever with the specified vector store type.

        Args:
            vector_store_type: 'faiss' or 'chroma'
            faiss_index_path: Path to save/load FAISS index
            chroma_dir: Directory for Chroma DB persistence
        """
        self.vector_store_type = vector_store_type.lower()
        self.faiss_index_path = faiss_index_path
        self.chroma_dir = chroma_dir

        if self.vector_store_type == "faiss":
            self.index = None
            if self.faiss_index_path and os.path.exists(self.faiss_index_path):
                self.index = faiss.read_index(self.faiss_index_path)
        elif self.vector_store_type == "chroma":
            settings = Settings(persist_directory=self.chroma_dir, chroma_db_impl="duckdb+parquet", anonymized_telemetry=False)
            self.client = chromadb.Client(settings)
            self.collection = self.client.get_or_create_collection(name="rag_collection")
        else:
            raise ValueError("Unsupported vector store type. Use 'faiss' or 'chroma'.")

    def build_index(self, embeddings: List[List[float]]):
        """
        Build the vector store index from embeddings.

        Args:
            embeddings: List of embeddings vectors.
        """
        if self.vector_store_type == "faiss":
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype("float32"))

            if self.faiss_index_path:
                faiss.write_index(self.index, self.faiss_index_path)

        elif self.vector_store_type == "chroma":
            # Store embeddings in Chroma collection with dummy ids
            ids = [str(i) for i in range(len(embeddings))]
            self.collection.add(
                documents=[""] * len(embeddings),  # Empty documents; actual docs stored externally
                embeddings=embeddings,
                ids=ids,
            )

    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        Retrieve the top_k most similar documents or IDs based on query embedding.

        Args:
            query_embedding: Embedding vector of query.
            top_k: Number of results to return.

        Returns:
            List of string IDs or document references.
        """
        if self.vector_store_type == "faiss":
            if self.index is None:
                raise RuntimeError("FAISS index is not built.")
            query_vec = np.array([query_embedding]).astype("float32")
            distances, indices = self.index.search(query_vec, top_k)
            return indices[0].tolist()

        elif self.vector_store_type == "chroma":
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return results["ids"][0]

def build_vector_store(vector_store_type: str, embeddings: List[List[float]], faiss_index_path=None, chroma_dir=None) -> Retriever:
    """
    Helper function to build and return a Retriever instance with the vector store loaded.

    Args:
        vector_store_type: 'faiss' or 'chroma'
        embeddings: List of embedding vectors.
        faiss_index_path: Optional FAISS index path.
        chroma_dir: Optional Chroma DB persistence directory.

    Returns:
        Retriever instance with built index.
    """
    retriever = Retriever(vector_store_type, faiss_index_path, chroma_dir)
    retriever.build_index(embeddings)
    return retriever


def retrieve_context(vector_store: Retriever, query: str, k: int = 5, id_to_doc: dict = None) -> List[str]:
    # Embed the query
    from utils.embedder import create_embedding_model

    embedder = create_embedding_model()
    query_embedding = embedder.encode([query])[0]

    ids = vector_store.retrieve(query_embedding, top_k=k)

    # Map IDs back to actual document chunks
    if id_to_doc is None:
        # fallback: just return empty or IDs
        return [f"Document chunk with ID: {idx}" for idx in ids]

    return [id_to_doc.get(str(idx), "") for idx in ids]
