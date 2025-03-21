"""
Agentis MCP - A flexible multi-agent framework with MCP server connectivity.
"""

__version__ = "0.1.0"

# Core components
from agentis_mcp.core.context import AgentContext

# MCP connectivity
from agentis_mcp.mcp.server_registry import ServerRegistry
from agentis_mcp.mcp.connection_manager import ConnectionManager
from agentis_mcp.mcp.aggregator import ServerAggregator, CompoundServer
from agentis_mcp.mcp.client_session import AgentisMCPClientSession

# Agent implementations
from agentis_mcp.agents.agent import Agent
from agentis_mcp.agents.llm_agent import LLMAgent, LLMConfig

# Configuration
from agentis_mcp.config import load_config, Settings

__all__ = [
    "AgentContext",
    "ServerRegistry",
    "ConnectionManager",
    "ServerAggregator",
    "CompoundServer",
    "AgentisMCPClientSession",
    "Agent",
    "LLMAgent",
    "LLMConfig",
    "load_config",
    "Settings",
]