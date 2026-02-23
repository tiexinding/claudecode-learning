from typing import Optional
from .base import LLMProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider

class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    SUPPORTED_PROVIDERS = {
        'claude': ClaudeProvider,
        'gemini': GeminiProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, api_key: str, model: str) -> Optional[LLMProvider]:
        """
        Create an LLM provider instance
        
        Args:
            provider_name: Name of the provider ('claude', 'gemini', etc.)
            api_key: API key for the provider
            model: Model name to use
            
        Returns:
            LLMProvider instance or None if provider not supported
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: {list(cls.SUPPORTED_PROVIDERS.keys())}")
        
        provider_class = cls.SUPPORTED_PROVIDERS[provider_name]
        return provider_class(api_key, model)
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported provider names"""
        return list(cls.SUPPORTED_PROVIDERS.keys())