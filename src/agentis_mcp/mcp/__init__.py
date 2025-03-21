"""
MCP connectivity for the Agentis MCP framework.

This module provides the components for connecting to MCP servers,
managing server connections, and routing tool calls to the appropriate servers.
"""

from .server_registry import ServerRegistry, InitHookCallable
from .connection_manager import ConnectionManager, ServerConnection
from .client_session import AgentisMCPClientSession
from .aggregator import ServerAggregator, CompoundServer
from .gen_client import gen_client, connect, disconnect

__all__ = [
    "ServerRegistry",
    "InitHookCallable",
    "ConnectionManager",
    "ServerConnection",
    "AgentisMCPClientSession",
    "ServerAggregator",
    "CompoundServer",
    "gen_client",
    "connect",
    "disconnect",
]