"""
Vector Storage and Retrieval System for HR Policy Chatbot
Handles embedding generation, vector storage, and semantic search using ChromaDB and OpenAI.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
import openai


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


def create_vector_store(
    documents: List[Document],
    api_key: str,
    embedding_model: str = "text-embedding-ada-002"
) -> Chroma:
    """
    Create a vector store from document chunks using Chroma and OpenAI embeddings.
    
    This function generates embeddings for each document chunk and stores them
    in an in-memory ChromaDB instance for semantic similarity search.
    
    Args:
        documents: List of Document objects to embed and store
        api_key: OpenAI API key for embedding generation
        embedding_model: OpenAI embedding model to use (default: text-embedding-ada-002)
        
    Returns:
        Chroma: Vector store instance containing embedded documents
        
    Raises:
        VectorStoreError: If vector store creation or embedding generation fails
    """
    if not documents or len(documents) == 0:
        raise VectorStoreError(
            "Cannot create vector store: No documents provided"
        )
    
    if not api_key or not isinstance(api_key, str):
        raise VectorStoreError(
            "Cannot create vector store: Invalid API key provided"
        )
    
    try:
        # Configure OpenAI embeddings with the specified model
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=embedding_model
        )
        
        # Create in-memory Chroma vector store from documents
        # Chroma will automatically generate embeddings for each document
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            # In-memory storage (no persist_directory specified)
        )
        
        return vector_store
        
    except openai.AuthenticationError as e:
        raise VectorStoreError(
            "Authentication failed: Invalid OpenAI API key\n"
            f"Error: {str(e)}\n"
            "Please check your API key at https://platform.openai.com/api-keys"
        ) from e
    
    except openai.RateLimitError as e:
        raise VectorStoreError(
            "Rate limit exceeded: Too many requests to OpenAI API\n"
            f"Error: {str(e)}\n"
            "Please wait a moment and try again, or check your API usage limits"
        ) from e
    
    except openai.APIConnectionError as e:
        raise VectorStoreError(
            "API connection failed: Unable to connect to OpenAI\n"
            f"Error: {str(e)}\n"
            "Please check your internet connection and try again"
        ) from e
    
    except openai.APIError as e:
        raise VectorStoreError(
            "OpenAI API error occurred during embedding generation\n"
            f"Error: {str(e)}\n"
            "Please try again later or contact support if the issue persists"
        ) from e
    
    except Exception as e:
        raise VectorStoreError(
            f"Failed to create vector store: {str(e)}\n"
            "An unexpected error occurred during vector store creation"
        ) from e



def get_retriever(
    vector_store: Chroma,
    k: int = 4,
    search_type: str = "similarity"
) -> VectorStoreRetriever:
    """
    Create a retriever from the vector store for semantic search.
    
    The retriever uses cosine similarity to find the most relevant document
    chunks for a given query.
    
    Args:
        vector_store: Chroma vector store instance
        k: Number of most relevant chunks to retrieve (default: 4)
        search_type: Type of search to perform (default: "similarity" for cosine similarity)
        
    Returns:
        VectorStoreRetriever: Configured retriever for semantic search
        
    Raises:
        VectorStoreError: If retriever creation fails
    """
    if vector_store is None:
        raise VectorStoreError(
            "Cannot create retriever: Vector store is None"
        )
    
    if k <= 0:
        raise VectorStoreError(
            f"Cannot create retriever: k must be positive (got {k})"
        )
    
    try:
        # Create retriever with specified parameters
        # Chroma uses cosine similarity by default for the "similarity" search type
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        return retriever
        
    except Exception as e:
        raise VectorStoreError(
            f"Failed to create retriever: {str(e)}"
        ) from e
