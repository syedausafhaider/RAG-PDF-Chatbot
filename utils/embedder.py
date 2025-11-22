from sentence_transformers import SentenceTransformer
from typing import List


def create_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return the embedding model.

    Args:
        model_name: Hugging Face model name or local path.

    Returns:
        SentenceTransformer embedding model instance.
    """
    model = SentenceTransformer(model_name)
    return model


def embed_documents(model: SentenceTransformer, documents: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text documents.

    Args:
        model: The sentence transformer model instance.
        documents: List of text chunks/documents to be embedded.

    Returns:
        List of embedding vectors (list of floats).
    """
    embeddings = model.encode(documents, convert_to_tensor=False, show_progress_bar=True)
    return embeddings