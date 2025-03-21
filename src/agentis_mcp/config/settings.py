"""
Settings models for the Agentis MCP framework.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

import yaml
from pydantic import BaseModel, Field, field_validator


class MCPServerRootSettings(BaseModel):
    """Settings for an MCP server root."""

    uri: str
    name: str
    server_uri_alias: Optional[str] = None


class MCPServerAuthSettings(BaseModel):
    """Settings for MCP server authentication."""

    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class MCPServerSettings(BaseModel):
    """Settings for an MCP server."""

    name: Optional[str] = None
    transport: str = "stdio"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    read_timeout_seconds: Optional[int] = None
    auth: Optional[MCPServerAuthSettings] = None
    roots: Optional[List[MCPServerRootSettings]] = None


class MCPSettings(BaseModel):
    """Settings for MCP configuration."""

    servers: Dict[str, MCPServerSettings] = Field(default_factory=dict)


class LLMProviderSettings(BaseModel):
    """Base settings for an LLM provider."""
    
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: Optional[str] = None
    

class OpenAISettings(LLMProviderSettings):
    """Settings for OpenAI."""
    
    api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    api_base: Optional[str] = Field(default="https://api.openai.com/v1", env="OPENAI_API_BASE")
    model: Optional[str] = Field(default="gpt-4o", env="OPENAI_MODEL")
    
    def __init__(self, **data):
        # Try to get API key from environment if not explicitly provided
        if "api_key" not in data:
            try:
                # Import here to avoid circular imports
                from ..utils.secrets import get_api_key
                api_key = get_api_key("openai")
                if api_key:
                    data["api_key"] = api_key
            except (ImportError, ValueError):
                # Fall back to environment variable handling in Field
                pass
        super().__init__(**data)


class AnthropicSettings(LLMProviderSettings):
    """Settings for Anthropic."""
    
    api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    api_base: Optional[str] = Field(default="https://api.anthropic.com/v1", env="ANTHROPIC_API_BASE")
    model: Optional[str] = Field(default="claude-3-7-sonnet-20250219", env="ANTHROPIC_MODEL")
    
    def __init__(self, **data):
        # Try to get API key from environment if not explicitly provided
        if "api_key" not in data:
            try:
                # Import here to avoid circular imports
                from ..utils.secrets import get_api_key
                api_key = get_api_key("anthropic")
                if api_key:
                    data["api_key"] = api_key
            except (ImportError, ValueError):
                # Fall back to environment variable handling in Field
                pass
        super().__init__(**data)


class AgentSettings(BaseModel):
    """Settings for an Agent."""

    name: str
    description: Optional[str] = None
    server_names: List[str] = Field(default_factory=list)
    connection_persistence: bool = False
    llm: Optional[Dict[str, Any]] = None


class WorkflowSettings(BaseModel):
    """Settings for workflow configuration."""

    enabled: bool = True
    parallelism: int = 1


class LoggingSettings(BaseModel):
    """Settings for logging configuration."""
    
    level: str = Field(default="info", env="LOG_LEVEL")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE")
    console: bool = True
    format: str = "json"


class Settings(BaseModel):
    """Root settings object for the Agentis MCP framework."""

    mcp: MCPSettings = Field(default_factory=MCPSettings)
    agents: Dict[str, AgentSettings] = Field(default_factory=dict)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    
    model_config = {
        "env_prefix": "AGENTIS_",
        "extra": "allow",
    }
    
    def __init__(self, **kwargs):
        # Gather environment variables
        for key, value in os.environ.items():
            if key.startswith("AGENTIS_") or key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", 
                                                   "OPENAI_API_BASE", "ANTHROPIC_API_BASE",
                                                   "OPENAI_MODEL", "ANTHROPIC_MODEL",
                                                   "LOG_LEVEL", "LOG_FILE"}:
                # Apply directly to relevant settings
                if key == "OPENAI_API_KEY":
                    kwargs.setdefault("openai", {}).setdefault("api_key", value)
                elif key == "ANTHROPIC_API_KEY":
                    kwargs.setdefault("anthropic", {}).setdefault("api_key", value)
                # Add other keys as needed
        
        super().__init__(**kwargs)


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load and validate the configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
            If None, look for 'agentis_mcp.config.yaml' in the current directory.

    Returns:
        Settings: Validated configuration object.
    """
    if config_path is None:
        # Look for config in current directory
        config_path = os.path.join(os.getcwd(), "agentis_mcp.config.yaml")
    
    config_data = {}
    
    # Load config if it exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
    
    # Load secrets if they exist
    secrets_path = Path(config_path).with_suffix(".secrets.yaml")
    if secrets_path.exists():
        with open(secrets_path, "r") as f:
            secrets_data = yaml.safe_load(f) or {}
        
        # Merge secrets into config
        _merge_dicts(config_data, secrets_data)
    
    # Environment variables override file settings
    env_config = _load_from_env()
    if env_config:
        _merge_dicts(config_data, env_config)
    
    return Settings.model_validate(config_data)


def _load_from_env() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dict with configuration loaded from environment variables.
    """
    config = {}
    
    # Configure LLM providers from environment variables
    _set_nested_dict(config, ["openai", "api_key"], os.environ.get("OPENAI_API_KEY"))
    _set_nested_dict(config, ["openai", "api_base"], os.environ.get("OPENAI_API_BASE"))
    _set_nested_dict(config, ["openai", "model"], os.environ.get("OPENAI_MODEL"))
    
    _set_nested_dict(config, ["anthropic", "api_key"], os.environ.get("ANTHROPIC_API_KEY"))
    _set_nested_dict(config, ["anthropic", "api_base"], os.environ.get("ANTHROPIC_API_BASE"))
    _set_nested_dict(config, ["anthropic", "model"], os.environ.get("ANTHROPIC_MODEL"))
    
    # Configure logging from environment variables
    _set_nested_dict(config, ["logging", "level"], os.environ.get("LOG_LEVEL"))
    _set_nested_dict(config, ["logging", "file_path"], os.environ.get("LOG_FILE"))
    
    # Process all AGENTIS_* environment variables
    for key, value in os.environ.items():
        if key.startswith("AGENTIS_"):
            # Convert AGENTIS_SECTION_SUBSECTION_KEY to section.subsection.key
            parts = key[8:].lower().split("_")
            path = []
            for part in parts:
                path.append(part)
            
            _set_nested_dict(config, path, value)
    
    return config


def _set_nested_dict(d: Dict[str, Any], path: List[str], value: Any) -> None:
    """
    Set a value in a nested dictionary based on a path.
    
    Args:
        d: Dictionary to set value in.
        path: List of keys defining the path.
        value: Value to set.
    """
    if value is None:
        return
    
    if len(path) == 1:
        d[path[0]] = value
        return
    
    if path[0] not in d:
        d[path[0]] = {}
    
    _set_nested_dict(d[path[0]], path[1:], value)


def _merge_dicts(target: Dict, source: Dict) -> None:
    """
    Recursively merge source dictionary into target dictionary.
    Values in source will override values in target.
    
    Args:
        target: Target dictionary to merge into.
        source: Source dictionary with values to merge.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_dicts(target[key], value)
        else:
            target[key] = value