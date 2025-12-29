"""
Document Processing Pipeline for HR Policy Chatbot
Handles loading and splitting PDF documents for vector embedding.
"""

import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


def load_single_document(pdf_path: str) -> List[Document]:
    """
    Load a single PDF document using PyPDFLoader.
    
    Args:
        pdf_path: Path to the PDF file to load
        
    Returns:
        List[Document]: List of Document objects with page content and metadata
        
    Raises:
        DocumentProcessingError: If the file cannot be loaded
    """
    # Validate file path exists
    if not os.path.exists(pdf_path):
        raise DocumentProcessingError(
            f"File not found: {pdf_path}\n"
            f"Please check that the file path is correct and the file exists."
        )
    
    # Validate it's a PDF file
    if not pdf_path.lower().endswith('.pdf'):
        raise DocumentProcessingError(
            f"Invalid file type: {pdf_path}\n"
            f"Only PDF files are supported."
        )
    
    try:
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Check if the PDF is empty
        if not documents or len(documents) == 0:
            raise DocumentProcessingError(
                f"Empty document: {pdf_path}\n"
                f"The PDF file contains no readable content."
            )
        
        # Metadata is automatically included by PyPDFLoader (source, page)
        return documents
        
    except DocumentProcessingError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Catch any other errors (corrupted PDF, parsing errors, etc.)
        raise DocumentProcessingError(
            f"Failed to load PDF: {pdf_path}\n"
            f"Error: {str(e)}\n"
            f"The file may be corrupted or in an unsupported format."
        ) from e


def load_documents(pdf_paths: List[str]) -> Tuple[List[Document], List[str], List[str]]:
    """
    Load multiple PDF documents from a list of file paths.
    
    Continues processing remaining files if one fails, collecting both
    successful and failed results.
    
    Args:
        pdf_paths: List of paths to PDF files to load
        
    Returns:
        Tuple containing:
        - List[Document]: All successfully loaded documents
        - List[str]: List of successfully loaded file paths
        - List[str]: List of error messages for failed files
    """
    all_documents = []
    successful_files = []
    error_messages = []
    
    for pdf_path in pdf_paths:
        try:
            documents = load_single_document(pdf_path)
            all_documents.extend(documents)
            successful_files.append(pdf_path)
        except DocumentProcessingError as e:
            error_messages.append(str(e))
        except Exception as e:
            error_messages.append(
                f"Unexpected error loading {pdf_path}: {str(e)}"
            )
    
    return all_documents, successful_files, error_messages



def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding generation.
    
    Uses RecursiveCharacterTextSplitter to maintain context across chunks
    by preserving natural text boundaries (paragraphs, sentences, etc.).
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk in characters (default: 1000)
        chunk_overlap: Number of characters to overlap between chunks (default: 200)
        
    Returns:
        List[Document]: List of split document chunks with preserved metadata
        
    Raises:
        DocumentProcessingError: If splitting fails
    """
    if not documents or len(documents) == 0:
        raise DocumentProcessingError(
            "Cannot split documents: No documents provided.\n"
            "Please ensure documents are loaded successfully before splitting."
        )
    
    # Validate chunk parameters
    if chunk_size <= 0:
        raise DocumentProcessingError(
            f"Invalid chunk_size: {chunk_size}. Chunk size must be positive."
        )
    
    if chunk_overlap < 0:
        raise DocumentProcessingError(
            f"Invalid chunk_overlap: {chunk_overlap}. Chunk overlap cannot be negative."
        )
    
    if chunk_overlap >= chunk_size:
        raise DocumentProcessingError(
            f"Invalid parameters: chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )
    
    try:
        # Create text splitter with specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Try to split on natural boundaries
        )
        
        # Split all documents
        split_docs = text_splitter.split_documents(documents)
        
        if not split_docs or len(split_docs) == 0:
            raise DocumentProcessingError(
                "Document splitting resulted in no chunks. "
                "Documents may be too short or contain no text."
            )
        
        return split_docs
        
    except DocumentProcessingError:
        raise
    except Exception as e:
        raise DocumentProcessingError(
            f"Failed to split documents: {str(e)}"
        ) from e
