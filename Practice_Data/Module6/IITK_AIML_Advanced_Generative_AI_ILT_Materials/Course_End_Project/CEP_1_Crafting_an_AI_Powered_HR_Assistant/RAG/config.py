"""
Configuration module for HR Policy Chatbot
Handles loading and validation of OpenAI API credentials from environment variables.
"""

import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


def load_openai_api_key() -> str:
    """
    Load OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key
        
    Raises:
        ConfigurationError: If the API key is not found in environment variables
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ConfigurationError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
            "You can create a .env file with: OPENAI_API_KEY=your_api_key_here"
        )
    
    return api_key


def validate_api_key(api_key: str) -> bool:
    """
    Validate that the API key is properly formatted.
    
    OpenAI API keys typically start with 'sk-' and contain alphanumeric characters.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if the API key appears to be valid
        
    Raises:
        ConfigurationError: If the API key format is invalid
    """
    if not api_key or not isinstance(api_key, str):
        raise ConfigurationError(
            "Invalid API key: API key must be a non-empty string"
        )
    
    # Remove whitespace
    api_key = api_key.strip()
    
    if len(api_key) == 0:
        raise ConfigurationError(
            "Invalid API key: API key cannot be empty or contain only whitespace"
        )
    
    # Check if API key starts with 'sk-' (standard OpenAI format)
    if not api_key.startswith("sk-"):
        raise ConfigurationError(
            "Invalid API key format: OpenAI API keys should start with 'sk-'\n"
            "Please check your API key at https://platform.openai.com/api-keys"
        )
    
    # Check minimum length (OpenAI keys are typically longer than 20 characters)
    if len(api_key) < 20:
        raise ConfigurationError(
            "Invalid API key format: API key appears to be too short\n"
            "Please verify your API key at https://platform.openai.com/api-keys"
        )
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^sk-[A-Za-z0-9_-]+$', api_key):
        raise ConfigurationError(
            "Invalid API key format: API key contains invalid characters\n"
            "API keys should only contain letters, numbers, hyphens, and underscores"
        )
    
    return True


def get_validated_api_key() -> str:
    """
    Load and validate the OpenAI API key.
    
    This is the main function to use for getting a validated API key.
    It combines loading from environment and validation.
    
    Returns:
        str: A validated OpenAI API key
        
    Raises:
        ConfigurationError: If the API key is missing or invalid
    """
    try:
        api_key = load_openai_api_key()
        validate_api_key(api_key)
        return api_key.strip()
    except ConfigurationError as e:
        # Re-raise with additional context
        raise ConfigurationError(
            f"Configuration error: {str(e)}\n\n"
            "To fix this:\n"
            "1. Get your API key from https://platform.openai.com/api-keys\n"
            "2. Create a .env file in the project root\n"
            "3. Add the line: OPENAI_API_KEY=your_api_key_here\n"
            "4. Make sure the .env file is not committed to version control"
        ) from e


# Initialize and validate API key on module import (optional - can be called explicitly instead)
def initialize_config() -> dict:
    """
    Initialize configuration and return config dictionary.
    
    Returns:
        dict: Configuration dictionary with validated settings
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = {
        "openai_api_key": get_validated_api_key(),
        "embedding_model": "text-embedding-ada-002",
        "llm_model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retriever_k": 4
    }
    
    return config
