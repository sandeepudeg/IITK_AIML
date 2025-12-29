"""
Question-Answering Engine for HR Policy Chatbot
Handles prompt template creation, QA chain setup, and question answering using LangChain and OpenAI.
"""

from typing import Dict, Any
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever


class QAEngineError(Exception):
    """Custom exception for QA engine operations."""
    pass


# Prompt template for HR Policy Chatbot
# Includes role definition, context and question placeholders, and clear instructions
PROMPT_TEMPLATE = """You are an HR assistant for Nestlé. Answer questions about HR policies based on the provided context.

Context: {context}

Question: {question}

Instructions:
- Provide clear and accurate answers based only on the context provided
- If the answer is not in the context, say "I don't have that information in the HR policy documents"
- Be professional and concise
- Cite specific policy sections when relevant

Answer:"""


def create_prompt_template() -> PromptTemplate:
    """
    Create the prompt template for the QA system.
    
    The template defines how the LLM should behave as an HR assistant,
    including instructions for handling missing information and maintaining
    a professional tone.
    
    Returns:
        PromptTemplate: Configured prompt template with context and question variables
    """
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    return prompt



def create_qa_chain(
    retriever: VectorStoreRetriever,
    api_key: str,
    model: str = "gpt-3.5-turbo",
    temperature: int = 0,
    max_tokens: int = 500
) -> RetrievalQA:
    """
    Create a RetrievalQA chain that integrates retriever, LLM, and prompt template.
    
    The chain uses the "stuff" chain type, which concatenates all retrieved
    documents into the prompt context.
    
    Args:
        retriever: VectorStoreRetriever for semantic search
        api_key: OpenAI API key for LLM access
        model: OpenAI model to use (default: gpt-3.5-turbo)
        temperature: Temperature for response generation (default: 0 for deterministic)
        max_tokens: Maximum tokens in response (default: 500)
        
    Returns:
        RetrievalQA: Configured QA chain ready for question answering
        
    Raises:
        QAEngineError: If QA chain creation fails
    """
    if retriever is None:
        raise QAEngineError(
            "Cannot create QA chain: Retriever is None"
        )
    
    if not api_key or not isinstance(api_key, str):
        raise QAEngineError(
            "Cannot create QA chain: Invalid API key provided"
        )
    
    try:
        # Configure ChatOpenAI LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create prompt template
        prompt = create_prompt_template()
        
        # Create RetrievalQA chain with "stuff" chain type
        # "stuff" concatenates all retrieved documents into the prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
        
    except Exception as e:
        raise QAEngineError(
            f"Failed to create QA chain: {str(e)}"
        ) from e



def answer_question(qa_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    Answer a question using the QA chain.
    
    This function retrieves relevant document chunks from the vector store,
    generates an answer using GPT-3.5 Turbo with the formatted prompt,
    and returns both the answer and source documents for transparency.
    
    Args:
        qa_chain: Configured RetrievalQA chain
        question: User's question about HR policies
        
    Returns:
        Dict containing:
        - 'result': Generated answer string
        - 'source_documents': List of Document objects used as context
        
    Raises:
        QAEngineError: If question answering fails
    """
    if qa_chain is None:
        raise QAEngineError(
            "Cannot answer question: QA chain is None"
        )
    
    if not question or not isinstance(question, str):
        raise QAEngineError(
            "Cannot answer question: Invalid question provided"
        )
    
    # Strip whitespace and validate question is not empty
    question = question.strip()
    if len(question) == 0:
        raise QAEngineError(
            "Cannot answer question: Question is empty"
        )
    
    try:
        # Invoke the QA chain with the question
        # The chain will:
        # 1. Retrieve relevant chunks from vector store
        # 2. Format the prompt with context and question
        # 3. Generate answer using GPT-3.5 Turbo
        # 4. Return answer and source documents
        response = qa_chain.invoke({"query": question})
        
        # Validate response
        if not response or not isinstance(response, dict):
            raise QAEngineError(
                "Invalid response from QA chain. Please try again."
            )
        
        # Check if we got a result
        if 'result' not in response:
            raise QAEngineError(
                "No answer was generated. Please try rephrasing your question."
            )
        
        return response
        
    except QAEngineError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Provide more specific error messages based on error type
        error_str = str(e).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            raise QAEngineError(
                "Rate limit exceeded. Please wait a moment and try again.\n"
                "If this persists, check your OpenAI API usage limits."
            ) from e
        elif "authentication" in error_str or "401" in error_str:
            raise QAEngineError(
                "Authentication failed. Your API key may be invalid or expired.\n"
                "Please check your API key at: https://platform.openai.com/api-keys"
            ) from e
        elif "connection" in error_str or "timeout" in error_str:
            raise QAEngineError(
                "Connection error. Please check your internet connection and try again."
            ) from e
        elif "context length" in error_str or "token" in error_str:
            raise QAEngineError(
                "Your question or the retrieved context is too long.\n"
                "Please try asking a shorter, more focused question."
            ) from e
        else:
            raise QAEngineError(
                f"Failed to answer question: {str(e)}\n"
                "An error occurred while processing your question. Please try again."
            ) from e
