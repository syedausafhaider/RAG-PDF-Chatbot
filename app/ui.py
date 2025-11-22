import sys
import os

# Add parent directory to sys.path to find config and utils
sys.path.append(os.path.abspath(".."))

import streamlit as st
from config import PDF_FILE_PATH, VECTOR_STORE_TYPE, FAISS_INDEX_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from utils.pdf_loader import load_pdf_texts
from utils.embedder import create_embedding_model, embed_documents
from utils.retriever import build_vector_store, retrieve_context
from utils.llm import load_llm_model
from graphs.rag_flow import RagFlowManager


def main():
    st.title("RAG PDF Chatbot")

    st.sidebar.header("Upload your PDF")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Save uploaded file locally
        os.makedirs(os.path.dirname(PDF_FILE_PATH), exist_ok=True)
        with open(PDF_FILE_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Loaded PDF: {uploaded_file.name}")

        # Load and split PDF text into chunks
        documents = load_pdf_texts(PDF_FILE_PATH)

        # Map document IDs to text chunks
        id_to_doc = {str(i): doc for i, doc in enumerate(documents)}

        # Create embedding model and embed documents
        embedder = create_embedding_model(EMBEDDING_MODEL_NAME)
        embeddings = embed_documents(embedder, documents)

        # Build or load vector store with embeddings
        vector_store = build_vector_store(
            VECTOR_STORE_TYPE,
            embeddings,
            faiss_index_path=FAISS_INDEX_PATH,
            chroma_dir=CHROMA_DB_DIR,
        )

        # Load LLM model
        llm = load_llm_model(LLM_MODEL_NAME)

        # Setup RAG manager
        rag_manager = RagFlowManager(vector_store, llm)

        # User input field for questions
        user_question = st.text_input("Ask a question related to the PDF:")

        if user_question:
            # Retrieve relevant context chunks by IDs and get corresponding text
            context = retrieve_context(
                vector_store,
                user_question,
                k=5,
                id_to_doc=id_to_doc,
            )

            # Generate answer based on query and retrieved context
            answer = rag_manager.generate_rag_answer(
                user_question,
                context,
            )

            st.markdown("**Answer:**")
            st.write(answer)
    else:
        st.info("Please upload a PDF to get started.")


if __name__ == "__main__":
    main()
