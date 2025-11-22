# RAG PDF Chatbot
A Retrieval-Augmented Generation (RAG) based PDF chatbot application using Python, Hugging Face Transformers, FAISS/Chroma for vector search, and Streamlit for web UI. This project allows you to upload any PDF, generate embeddings from its text, and perform question answering with large language models over the PDF content.

---

## Features
- **PDF text loading and chunking**
- **Embeddings generation with Sentence Transformers**
- **Vector similarity search using FAISS or Chroma DB**
- **Integration with local or remote Large Language Models (LLMs)** for answer generation
- **Command-line and Streamlit-based interactive user interfaces**
- **Supports swapping between small and large language models** for faster local testing or production-grade usage

---

## Tech Stack
- **Python 3.12**
- **Hugging Face Transformers**
- **Sentence Transformers** (`sentence-transformers/all-MiniLM-L6-v2`)
- **FAISS or Chroma** for vector store
- **Streamlit** for the web UI
- **PyTorch** for model inference

---

## Getting Started

### Prerequisites
- **Python 3.7+**
- **GPU recommended** for LLM inference (CPU fallback available)

Install dependencies (preferably in a virtual environment):
```bash
pip install -r requirements.txt
```

---

## Configuration
Edit `config.py` to set paths and model names, for example:
```python
PDF_FILE_PATH = "data/sample.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_TYPE = "faiss"  # or "chroma"
FAISS_INDEX_PATH = "vector_db/faiss_index.index"
CHROMA_DB_DIR = "vector_db/chroma"
LLM_MODEL_NAME = "distilgpt2"  # small model for learning, replace with larger when ready
MAX_CONTEXT_TOKENS = 1024
TEMPERATURE = 0.7
```

---

## Running the Chatbot
### CLI Mode
```bash
python main.py
```
Interact with the chatbot via the terminal.

### Web UI
Run the Streamlit app:
```bash
streamlit run app/ui.py
```
Open your browser at **http://localhost:8501**.

---

## Project Structure
```
rag_pdf_chatbot/
├── app/
│   └── ui.py            # Streamlit web UI
├── graphs/
│   └── rag_flow.py      # RAG orchestration code
├── utils/
│   ├── embedder.py      # Embeddings model and helpers
│   ├── llm.py           # LLM loading and generation logic
│   ├── pdf_loader.py    # PDF loading and text chunking
│   ├── prompt.py        # Prompt construction helpers
│   └── retriever.py     # Vector store and retrieval helpers
├── config.py            # Configuration constants
├── main.py              # CLI chatbot entry point
├── requirements.txt     # Python package dependencies
└── data/
    └── sample.pdf       # Sample PDF for testing
```

---

## Usage Notes
- For quick experimentation, start with a small LLM model like **distilgpt2** to reduce resource requirements.
- To work with larger models (e.g., **Vicuna** or **LLaMA**), ensure you have sufficient GPU memory and download access.
- Vector stores speed up retrieval and can be persisted between runs.
- Prompts are designed to answer **strictly based on retrieved context**, improving factual accuracy.

---

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repo and open a pull request.

---

## License
This project is licensed under the **MIT License**.

---

## Acknowledgements
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Streamlit

