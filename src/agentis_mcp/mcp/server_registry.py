"""
Server registry for managing MCP server configurations and initialization.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Callable, Dict, AsyncGenerator, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
    get_default_environment,
)
from mcp.client.sse import sse_client

from agentis_mcp.config import (
    Settings,
    MCPServerAuthSettings,
    MCPServerSettings,
)
from agentis_mcp.utils.logging import get_logger

logger = get_logger(__name__)

InitHookCallable = Callable[[ClientSession, Optional[MCPServerAuthSettings]], None]
"""
A callback function that is invoked after MCP server initialization.

Args:
    session: The client session for the server connection.
    auth: The authentication configuration for the server.
"""


class ServerRegistry:
    """
    Manages server configurations and initialization logic.

    The ServerRegistry class is responsible for loading server configurations,
    registering initialization hooks, initializing servers, and executing
    post-initialization hooks dynamically.
    """

    def __init__(self, config: Settings):
        """
        Initialize the ServerRegistry with configuration.

        Args:
            config: The Settings object containing the server configurations.
        """
        self.registry = config.mcp.servers
        self.init_hooks: Dict[str, InitHookCallable] = {}
        
        # Initialize the connection manager
        from agentis_mcp.mcp.connection_manager import ConnectionManager
        self.connection_manager = ConnectionManager(self)

    @asynccontextmanager
    async def start_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Starts a server based on its configuration.

        Args:
            server_name: The name of the server to start.
            client_session_factory: Factory for creating client sessions.

        Yields:
            ClientSession: The client session connected to the server.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]

        read_timeout = (
            timedelta(seconds=config.read_timeout_seconds)
            if config.read_timeout_seconds
            else None
        )

        if config.transport == "stdio":
            if not config.command or not config.args:
                raise ValueError(
                    f"Command and args are required for stdio transport: {server_name}"
                )

            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env={**get_default_environment(), **(config.env or {})},
            )

            async with stdio_client(server_params) as (read_stream, write_stream):
                session = client_session_factory(
                    read_stream,
                    write_stream,
                    read_timeout,
                )
                async with session:
                    logger.info(
                        f"{server_name}: Connected to server using stdio transport."
                    )
                    try:
                        yield session
                    finally:
                        logger.debug(f"{server_name}: Closed session to server")

        elif config.transport == "sse":
            if not config.url:
                raise ValueError(f"URL is required for SSE transport: {server_name}")

            async with sse_client(config.url) as (read_stream, write_stream):
                session = client_session_factory(
                    read_stream,
                    write_stream,
                    read_timeout,
                )
                async with session:
                    logger.info(
                        f"{server_name}: Connected to server using SSE transport."
                    )
                    try:
                        yield session
                    finally:
                        logger.debug(f"{server_name}: Closed session to server")

        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
        init_hook: Optional[InitHookCallable] = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Initialize a server based on its configuration.
        After initialization, also calls any registered or provided initialization hook.

        Args:
            server_name: The name of the server to initialize.
            client_session_factory: Factory for creating client sessions.
            init_hook: Optional initialization hook function to call after initialization.

        Yields:
            ClientSession: The initialized client session.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]

        async with self.start_server(
            server_name, client_session_factory=client_session_factory
        ) as session:
            try:
                logger.info(f"{server_name}: Initializing server...")
                await session.initialize()
                logger.info(f"{server_name}: Initialized.")

                initialization_callback = (
                    init_hook
                    if init_hook is not None
                    else self.init_hooks.get(server_name)
                )

                if initialization_callback:
                    logger.info(f"{server_name}: Executing init hook")
                    initialization_callback(session, config.auth)

                logger.info(f"{server_name}: Up and running!")
                yield session
            finally:
                logger.info(f"{server_name}: Ending server session.")

    def register_init_hook(self, server_name: str, hook: InitHookCallable) -> None:
        """
        Register an initialization hook for a specific server.

        Args:
            server_name: The name of the server.
            hook: The initialization function to register.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        self.init_hooks[server_name] = hook