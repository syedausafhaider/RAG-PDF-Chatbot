from typing import List

class RagFlowManager:
    def __init__(self, vector_store, llm):
        """
        Initialize the RAG flow manager with a vector store and LLM model.
        
        Args:
            vector_store: The retrieval vector store instance (FAISS, Chroma, etc.)
            llm: The loaded language model instance for generation
        """
        self.vector_store = vector_store
        self.llm = llm

    def generate_rag_answer(
        self,
        query: str,
        context: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate an answer to a query using retrieved context and the LLM.
        
        Args:
            query: User question string.
            context: List of relevant text chunks retrieved from vector store.
            max_tokens: Max tokens for LLM generation.
            temperature: Sampling temperature for deterministic or creative responses.
        
        Returns:
            Generated answer string from the LLM.
        """
        # Build the prompt for the LLM, enforcing strict RAG constraints
        combined_context = "\n\n".join(context)
        prompt = (
            f"You are an AI assistant. Only use the following context to answer the question.\n"
            f"If the answer is not contained in the context, respond with 'I don't know.'\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        # Call the LLM with the prompt and generation params (assumes llm has a generate method)
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.strip()
