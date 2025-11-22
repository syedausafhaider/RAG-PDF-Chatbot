from typing import List


def build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Build the prompt for the LLM using the user query and retrieved context chunks.

    Args:
        query: The user question string.
        context_chunks: List of text chunks retrieved from the vector database.

    Returns:
        A well-formatted string prompt for the LLM.
    """
    combined_context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful AI assistant. Use only the context below to answer the question. "
        "If the answer is not contained within the context, respond with 'I don't know.'\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt
