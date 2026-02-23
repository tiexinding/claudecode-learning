import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # LLM Provider settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "claude")  # claude, gemini
    
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    
    # Google Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()

# Helper functions for provider configuration
def get_llm_config():
    """Get LLM configuration based on selected provider"""
    provider = config.LLM_PROVIDER.lower()
    
    if provider == "claude":
        return {
            "provider": "claude",
            "api_key": config.ANTHROPIC_API_KEY,
            "model": config.ANTHROPIC_MODEL
        }
    elif provider == "gemini":
        return {
            "provider": "gemini", 
            "api_key": config.GEMINI_API_KEY,
            "model": config.GEMINI_MODEL
        }
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def validate_llm_config():
    """Validate that required API keys are present for the selected provider"""
    provider = config.LLM_PROVIDER.lower()
    
    if provider == "claude" and not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is required when using Claude provider")
    elif provider == "gemini" and not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when using Gemini provider")


