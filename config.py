import os
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
MODEL_DIR = os.path.join(BASE_DIR, "models", "local_llm")

# PDF file path (example)
PDF_FILE_PATH = os.path.join(DATA_DIR, "sample.pdf")

# Vector store settings
VECTOR_STORE_TYPE = "faiss"  # Options: "faiss", "chroma"
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index.index")
CHROMA_DB_DIR = os.path.join(VECTOR_DB_DIR, "chroma_db")

# Embeddings model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local LLM model name or path
LLM_MODEL_NAME = "distilgpt2"


  # Example: Llama 2 7B Hugging Face repo

# Prompt config
MAX_CONTEXT_TOKENS = 2048
TEMPERATURE = 0.0  # Deterministic answers preferred in RAG

# Logging config
LOGGING_LEVEL = "INFO"

# Environment variables for external APIs if any (not used here since local LLM)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

