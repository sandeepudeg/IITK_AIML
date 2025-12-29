"""
Gradio User Interface for HR Policy Chatbot
Provides web interface for document upload and question answering.
"""

import os
from typing import List, Tuple, Optional, Any
import gradio as gr

from document_processor import load_documents, split_documents, DocumentProcessingError
from vector_store import create_vector_store, get_retriever, VectorStoreError
from qa_engine import create_qa_chain, answer_question, QAEngineError


class InterfaceError(Exception):
    """Custom exception for interface operations."""
    pass


# Global variables to store state
vector_store: Optional[Any] = None
qa_chain: Optional[Any] = None
api_key: str = ""


def upload_documents(files: List[gr.File], api_key_param: str, progress=gr.Progress()) -> str:
    """
    Handle multiple PDF file uploads and add them to the vector store.
    
    Validates file types (.pdf only) and file sizes (max 10MB per file),
    processes uploaded files, and adds them to the vector store.
    
    Args:
        files: List of uploaded file objects from Gradio
        api_key_param: OpenAI API key for embedding generation
        progress: Gradio Progress tracker for loading indicators
        
    Returns:
        str: Status message showing successfully processed and failed files
    """
    global vector_store, qa_chain, api_key
    
    # Validate inputs
    if not files or len(files) == 0:
        return "❌ **No files uploaded**\n\nPlease select one or more PDF files to upload."
    
    if not api_key_param or not isinstance(api_key_param, str) or len(api_key_param.strip()) == 0:
        return "❌ **API key is required**\n\nPlease enter your OpenAI API key in the field above.\n\n💡 Get your API key at: https://platform.openai.com/api-keys"
    
    # Store API key for later use
    api_key = api_key_param.strip()
    
    # Validate API key format
    try:
        from config import validate_api_key
        validate_api_key(api_key)
    except Exception as e:
        return f"❌ **Invalid API key**\n\n{str(e)}\n\n💡 Get your API key at: https://platform.openai.com/api-keys"
    
    progress(0, desc="Validating files...")
    
    # Validate file types and sizes
    valid_files = []
    validation_errors = []
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    
    for file in files:
        file_path = file.name
        file_name = os.path.basename(file_path)
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            validation_errors.append(f"❌ {file_name}: Invalid file type (only .pdf files are accepted)")
            continue
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE:
                size_mb = file_size / (1024 * 1024)
                validation_errors.append(f"❌ {file_name}: File too large ({size_mb:.1f}MB, max 10MB)")
                continue
        except Exception as e:
            validation_errors.append(f"❌ {file_name}: Could not check file size - {str(e)}")
            continue
        
        valid_files.append(file_path)
    
    if len(valid_files) == 0:
        error_msg = "❌ **No valid files to process**\n\n" + "\n".join(validation_errors)
        error_msg += "\n\n💡 **Requirements:**\n• File type: PDF only\n• Max size: 10MB per file"
        return error_msg
    
    # Process valid files
    try:
        progress(0.2, desc=f"Loading {len(valid_files)} PDF file(s)...")
        
        # Load documents
        documents, successful_files, load_errors = load_documents(valid_files)
        
        if len(documents) == 0:
            error_msg = "❌ **No documents could be loaded**\n\n"
            if load_errors:
                error_msg += "**Errors:**\n" + "\n".join(load_errors)
            error_msg += "\n\n💡 **Troubleshooting:**\n• Ensure PDFs are not corrupted\n• Check that PDFs contain readable text\n• Try opening the PDFs in a PDF reader first"
            return error_msg
        
        progress(0.4, desc=f"Splitting {len(documents)} page(s) into chunks...")
        
        # Split documents into chunks
        chunks = split_documents(documents)
        
        progress(0.6, desc=f"Generating embeddings for {len(chunks)} chunk(s)...")
        
        # Create or update vector store
        if vector_store is None:
            # Create new vector store
            vector_store = create_vector_store(chunks, api_key)
        else:
            # Add to existing vector store
            vector_store.add_documents(chunks)
        
        progress(0.9, desc="Creating QA chain...")
        
        # Create or update QA chain
        retriever = get_retriever(vector_store)
        qa_chain = create_qa_chain(retriever, api_key)
        
        # Build status message
        status_parts = []
        
        if successful_files:
            status_parts.append(f"✅ Successfully processed {len(successful_files)} file(s):")
            for file_path in successful_files:
                file_name = os.path.basename(file_path)
                status_parts.append(f"  • {file_name}")
        
        if validation_errors:
            status_parts.append(f"\n⚠️ Validation errors ({len(validation_errors)}):")
            status_parts.extend(validation_errors)
        
        if load_errors:
            status_parts.append(f"\n⚠️ Processing errors ({len(load_errors)}):")
            for error in load_errors:
                status_parts.append(f"  • {error}")
        
        status_parts.append(f"\n📊 **Summary:**")
        status_parts.append(f"  • Total pages processed: {len(documents)}")
        status_parts.append(f"  • Total chunks created: {len(chunks)}")
        status_parts.append("\n✅ **Ready to answer questions!**")
        
        progress(1.0, desc="Complete!")
        
        return "\n".join(status_parts)
        
    except (DocumentProcessingError, VectorStoreError, QAEngineError) as e:
        error_msg = f"❌ **Error processing documents**\n\n{str(e)}"
        error_msg += "\n\n💡 **What to do:**\n• Check the error message above\n• Verify your API key is correct\n• Ensure you have internet connectivity\n• Try uploading fewer or smaller files"
        return error_msg
    except Exception as e:
        error_msg = f"❌ **Unexpected error occurred**\n\n{str(e)}"
        error_msg += "\n\n💡 **What to do:**\n• Try uploading the files again\n• Check that your PDFs are not corrupted\n• Restart the application if the problem persists"
        return error_msg



def chatbot_interface(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Process user questions and generate answers using the QA chain.
    
    This function integrates with the QA chain to generate answers and
    maintains conversation history for context.
    
    Args:
        message: User's question
        history: List of (user_message, bot_response) tuples representing conversation history
        
    Returns:
        str: Formatted response for display in chat window
    """
    global qa_chain
    
    # Validate that documents have been uploaded
    if qa_chain is None:
        return "⚠️ **Please upload PDF documents first**\n\nUse the upload section above to add HR policy documents before asking questions."
    
    # Validate message
    if not message or not isinstance(message, str):
        return "⚠️ **Invalid question format**\n\nPlease enter a text question."
    
    message = message.strip()
    
    # Check for empty or very short queries
    if len(message) == 0:
        return "⚠️ **Empty question**\n\nPlease enter a question about the HR policies."
    
    if len(message) < 3:
        return "⚠️ **Question too short**\n\nPlease provide a more detailed question (at least 3 characters)."
    
    # Check for malformed queries (only special characters, no letters)
    if not any(c.isalnum() for c in message):
        return "⚠️ **Invalid question**\n\nYour question should contain letters or numbers. Please rephrase your question."
    
    try:
        # Answer the question using the QA chain
        response = answer_question(qa_chain, message)
        
        # Extract the answer from the response
        answer = response.get('result', 'No answer generated')
        
        # Check if answer indicates information not found
        if not answer or answer.strip() == "":
            return "⚠️ **No answer generated**\n\nThe system couldn't generate an answer. Please try rephrasing your question."
        
        # Optionally include source information
        source_docs = response.get('source_documents', [])
        if source_docs:
            # Add source information to the answer
            sources = set()
            for doc in source_docs:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                sources.add(f"{os.path.basename(source)} (page {page})")
            
            if sources:
                answer += f"\n\n📚 **Sources:** {', '.join(sorted(sources))}"
        
        return answer
        
    except QAEngineError as e:
        error_msg = f"❌ **Error processing question**\n\n{str(e)}"
        error_msg += "\n\n💡 **What to do:**\n• Try rephrasing your question\n• Make sure your question is clear and specific\n• Check your internet connection"
        return error_msg
    except Exception as e:
        error_msg = f"❌ **Unexpected error occurred**\n\n{str(e)}"
        error_msg += "\n\n💡 **What to do:**\n• Try asking your question again\n• Restart the application if the problem persists"
        return error_msg



def create_interface() -> gr.Blocks:
    """
    Build the Gradio interface layout with file upload and chat components.
    
    Creates a Blocks layout with:
    - File upload component at the top
    - API key input field
    - Upload status messages
    - ChatInterface for conversation display
    - Text input field for questions
    - Submit and clear buttons
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(title="HR Policy Chatbot - Nestlé", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 HR Policy Chatbot - Nestlé")
        gr.Markdown("Upload HR policy PDF documents and ask questions about them using AI-powered search.")
        
        # File upload section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Step 1: Upload Documents")
                gr.Markdown("*Required: Provide your OpenAI API key and upload PDF files*")
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="Enter your OpenAI API key (sk-...)",
                    type="password",
                    lines=1,
                    info="Get your API key at: https://platform.openai.com/api-keys"
                )
                file_upload = gr.File(
                    label="Upload PDF Files (Max 10MB each)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )
                upload_button = gr.Button("📤 Upload and Process", variant="primary", size="lg")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=10,
                    interactive=False,
                    placeholder="Upload status will appear here...\n\n💡 Tip: You can upload multiple PDF files at once.",
                    show_copy_button=False
                )
        
        gr.Markdown("---")
        
        # Chat interface section
        gr.Markdown("### 💬 Step 2: Ask Questions")
        gr.Markdown("*After uploading documents, ask questions about the HR policies*")
        
        chatbot = gr.Chatbot(
            label="Conversation History",
            height=450,
            show_label=True,
            show_copy_button=True,
            avatar_images=(None, "🤖")
        )
        
        with gr.Row():
            message_input = gr.Textbox(
                label="Your Question",
                placeholder="Example: What is the vacation policy? How many sick days do employees get?",
                lines=2,
                scale=4,
                show_label=False
            )
            submit_button = gr.Button("Send", variant="primary", scale=1, size="lg")
        
        with gr.Row():
            clear_button = gr.Button("🗑️ Clear Conversation", size="sm")
        
        gr.Markdown("---")
        gr.Markdown("""
        ### 💡 Tips for Best Results
        - **Be specific**: Ask clear, focused questions about HR policies
        - **One topic at a time**: Break complex questions into simpler ones
        - **Check sources**: Review the source documents cited in answers
        - **Upload first**: Make sure to upload PDF documents before asking questions
        """)
        
        gr.Markdown("---")
        gr.Markdown("⚠️ **Note:** This chatbot answers based only on uploaded documents. Answers are generated by AI and should be verified with official HR policies.")
        
        # Wire up the upload functionality
        upload_button.click(
            fn=upload_documents,
            inputs=[file_upload, api_key_input],
            outputs=upload_status
        )
        
        # Wire up the chat functionality
        def respond(message, chat_history):
            """Handle chat interaction."""
            bot_response = chatbot_interface(message, chat_history)
            chat_history.append((message, bot_response))
            return "", chat_history
        
        submit_button.click(
            fn=respond,
            inputs=[message_input, chatbot],
            outputs=[message_input, chatbot]
        )
        
        message_input.submit(
            fn=respond,
            inputs=[message_input, chatbot],
            outputs=[message_input, chatbot]
        )
        
        # Wire up the clear functionality
        clear_button.click(
            fn=lambda: [],
            inputs=None,
            outputs=chatbot
        )
    
    return interface



def launch_interface(share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    """
    Launch the Gradio interface server.
    
    Starts the Gradio web server with the specified configuration for local deployment.
    
    Args:
        share: Whether to create a public link (default: False for local deployment)
        server_name: Server hostname (default: 127.0.0.1 for localhost)
        server_port: Server port (default: 7860)
    """
    interface = create_interface()
    
    print("🚀 Launching HR Policy Chatbot interface...")
    print(f"📍 Server will be available at: http://{server_name}:{server_port}")
    print("⚠️  Remember to upload PDF documents before asking questions!")
    
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )


if __name__ == "__main__":
    # Launch interface when run directly
    launch_interface()
