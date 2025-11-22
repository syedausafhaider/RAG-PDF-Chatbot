import pdfplumber
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter





def load_pdf_texts(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Load text from a PDF file and split it into chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks extracted from the PDF.
    """
    full_text = ""

    # Extract all text from the PDF with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # Use LangChain's RecursiveCharacterTextSplitter for chunking text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(full_text)

    return chunks
