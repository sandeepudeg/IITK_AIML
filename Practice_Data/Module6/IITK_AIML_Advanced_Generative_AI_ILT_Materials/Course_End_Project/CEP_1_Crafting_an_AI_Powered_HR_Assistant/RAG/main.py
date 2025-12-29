"""
HR Policy Chatbot - Main Application Entry Point
Initializes configuration, validates API setup, and launches the Gradio web interface.
"""

import sys
from config import get_validated_api_key, ConfigurationError, initialize_config
from gradio_interface import launch_interface


def main():
    """
    Main entry point for the HR Policy Chatbot application.
    
    This function:
    1. Initializes configuration and validates API key
    2. Displays startup information
    3. Launches the Gradio interface
    
    Exits with error code 1 if configuration validation fails.
    """
    print("=" * 60)
    print("🤖 HR Policy Chatbot - Nestlé")
    print("=" * 60)
    print()
    
    # Step 1: Validate API key before launching
    print("🔑 Validating OpenAI API configuration...")
    try:
        api_key = get_validated_api_key()
        print("✅ API key validated successfully")
        print()
    except ConfigurationError as e:
        print("❌ Configuration Error:")
        print(str(e))
        print()
        print("Application cannot start without a valid API key.")
        sys.exit(1)
    
    # Step 2: Initialize full configuration
    print("⚙️  Initializing configuration...")
    try:
        config = initialize_config()
        print(f"✅ Configuration loaded:")
        print(f"   - LLM Model: {config['llm_model']}")
        print(f"   - Embedding Model: {config['embedding_model']}")
        print(f"   - Chunk Size: {config['chunk_size']}")
        print(f"   - Retriever K: {config['retriever_k']}")
        print()
    except ConfigurationError as e:
        print("❌ Configuration Error:")
        print(str(e))
        sys.exit(1)
    
    # Step 3: Launch Gradio interface
    print("🚀 Launching Gradio interface...")
    print("📝 Instructions:")
    print("   1. Upload one or more HR policy PDF files")
    print("   2. Wait for processing to complete")
    print("   3. Start asking questions about the policies")
    print()
    print("⚠️  Note: The interface will open in your default web browser")
    print("=" * 60)
    print()
    
    try:
        # Launch interface with default settings (localhost:7860)
        launch_interface(
            share=False,  # Local deployment only
            server_name="127.0.0.1",
            server_port=7860
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching interface: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
