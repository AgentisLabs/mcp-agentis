"""
Secret management utilities for the Agentis MCP framework.

This module provides functions for securely accessing API keys and other secrets,
leveraging environment variables and .env files for local development.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    # Provide a fallback if dotenv is not installed
    def load_dotenv(dotenv_path=None):
        print("Warning: python-dotenv not installed. Environment variables from .env files will not be loaded.")
        print("Install with: pip install python-dotenv")
        return False

# Paths to check for .env files, in order of precedence
ENV_PATHS = [
    Path.cwd() / ".env",                      # Project root .env file
    Path.cwd() / ".secrets.env",              # Alternative secrets file
    Path.home() / ".agentis_mcp" / ".env",    # User-level config
]

# Load environment variables from .env file if it exists
for env_path in ENV_PATHS:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret from environment variables with fallback.
    
    Args:
        key: The environment variable name containing the secret
        default: Default value if the secret is not found
        
    Returns:
        The secret value or default if not found
    """
    return os.environ.get(key, default)


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider.
    
    Args:
        provider: Provider name ("openai", "anthropic", etc.)
        
    Returns:
        The API key for the specified provider or None if not found
    
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    
    if provider == "openai":
        return get_secret("OPENAI_API_KEY")
    elif provider == "anthropic":
        return get_secret("ANTHROPIC_API_KEY")
    elif provider == "brave":
        return get_secret("BRAVE_API_KEY")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def set_api_key(provider: str, api_key: str) -> None:
    """
    Set API key for a specific provider in the current environment.
    Note: This does not persist the key to the .env file.
    
    Args:
        provider: Provider name ("openai", "anthropic", etc.)
        api_key: The API key to set
    
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif provider == "brave":
        os.environ["BRAVE_API_KEY"] = api_key
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_env_template(output_path: str = ".env.example") -> None:
    """
    Generate a template .env file with placeholders for secrets.
    
    Args:
        output_path: Path where the template file should be created
    """
    template = """# Agentis MCP Secrets
# Save this file as .env in the project root or ~/.agentis_mcp/.env

# OpenAI API Configuration
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# Anthropic API Configuration
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_API_BASE=https://api.anthropic.com/v1
ANTHROPIC_MODEL=claude-3-7-sonnet-20250219

# Other API keys
BRAVE_API_KEY=...
"""
    with open(output_path, "w") as f:
        f.write(template)
    print(f"Created template file at {output_path}")


if __name__ == "__main__":
    # If run directly, generate a template .env file
    generate_env_template()