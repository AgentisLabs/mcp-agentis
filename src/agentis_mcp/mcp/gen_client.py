"""
Client generation utilities for connecting to MCP servers.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Callable, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession

from agentis_mcp.utils.logging import get_logger
from agentis_mcp.mcp.client_session import AgentisMCPClientSession

logger = get_logger(__name__)


@asynccontextmanager
async def gen_client(
    server_name: str,
    server_registry,
    client_session_factory: Callable[
        [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
        ClientSession,
    ] = AgentisMCPClientSession,
) -> AsyncGenerator[ClientSession, None]:
    """
    Create a temporary client session to the specified server.
    
    Args:
        server_name: Name of the server to connect to.
        server_registry: Registry containing server configurations.
        client_session_factory: Factory function for creating client sessions.
        
    Yields:
        ClientSession: A connected client session.
        
    Raises:
        ValueError: If server registry is not provided.
    """
    if not server_registry:
        raise ValueError(
            "Server registry not found. Please specify one on this method, or in the context."
        )

    async with server_registry.initialize_server(
        server_name=server_name,
        client_session_factory=client_session_factory,
    ) as session:
        yield session


async def connect(
    server_name: str,
    server_registry,
    client_session_factory: Callable[
        [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
        ClientSession,
    ] = AgentisMCPClientSession,
) -> ClientSession:
    """
    Create a persistent client session to the specified server.
    
    Args:
        server_name: Name of the server to connect to.
        server_registry: Registry containing server configurations.
        client_session_factory: Factory function for creating client sessions.
        
    Returns:
        ClientSession: A connected client session.
        
    Raises:
        ValueError: If server registry is not provided.
    """
    if not server_registry:
        raise ValueError(
            "Server registry not found. Please specify one on this method, or in the context."
        )

    server_connection = await server_registry.connection_manager.get_server(
        server_name=server_name,
        client_session_factory=client_session_factory,
    )

    return server_connection.session


async def disconnect(
    server_name: Optional[str],
    server_registry,
) -> None:
    """
    Disconnect from the specified server or all servers.
    
    Args:
        server_name: Name of the server to disconnect from, or None to disconnect from all.
        server_registry: Registry containing server configurations.
        
    Raises:
        ValueError: If server registry is not provided.
    """
    if not server_registry:
        raise ValueError(
            "Server registry not found. Please specify one on this method, or in the context."
        )

    if server_name:
        await server_registry.connection_manager.disconnect_server(
            server_name=server_name
        )
    else:
        await server_registry.connection_manager.disconnect_all()