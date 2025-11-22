import logging
import sys

from config import (
    PDF_FILE_PATH,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_TYPE,
    FAISS_INDEX_PATH,
    CHROMA_DB_DIR,
    LLM_MODEL_NAME,
    MAX_CONTEXT_TOKENS,
    TEMPERATURE,
)
from utils.pdf_loader import load_pdf_texts
from utils.embedder import create_embedding_model, embed_documents
from utils.retriever import build_vector_store, retrieve_context
from utils.llm import load_llm_model
from graphs.rag_flow import RagFlowManager


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    setup_logging()
    logging.info("Starting RAG PDF Chatbot")

    # Load PDF and split text
    logging.info(f"Loading PDF from: {PDF_FILE_PATH}")
    documents = load_pdf_texts(PDF_FILE_PATH)

    # Create a mapping from doc ID (string) to actual text chunk
    id_to_doc = {str(i): doc for i, doc in enumerate(documents)}

    # Create embeddings model
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = create_embedding_model(EMBEDDING_MODEL_NAME)

    # Embed documents and build vector DB
    logging.info("Embedding documents and setting up vector store")
    embeddings = embed_documents(embedder, documents)
    vector_store = build_vector_store(
        VECTOR_STORE_TYPE, embeddings, FAISS_INDEX_PATH, CHROMA_DB_DIR
    )

    # Load local LLM model
    logging.info(f"Loading LLM model: {LLM_MODEL_NAME}")
    llm = load_llm_model(LLM_MODEL_NAME)

    # Initialize RAG flow manager
    rag_manager = RagFlowManager(vector_store, llm)

    logging.info("Chatbot ready! Type your questions or 'exit' to quit.")

    # Simple CLI loop for demo
    while True:
        user_query = input("\nQuestion: ")
        if user_query.strip().lower() == "exit":
            logging.info("Exiting chatbot.")
            break

        # Retrieve relevant context and generate answer using actual text chunks
        context = retrieve_context(vector_store, user_query, k=5, id_to_doc=id_to_doc)

        answer = rag_manager.generate_rag_answer(
            query=user_query, context=context, max_tokens=MAX_CONTEXT_TOKENS, temperature=TEMPERATURE
        )
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
