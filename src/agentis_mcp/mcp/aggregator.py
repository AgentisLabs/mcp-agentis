"""
Server aggregation for combining multiple MCP servers.
"""

from asyncio import Lock, gather
from typing import List, Dict, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field
from mcp.client.session import ClientSession
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, Tool, TextContent

from agentis_mcp.utils.logging import get_logger
from agentis_mcp.mcp.client_session import AgentisMCPClientSession
from agentis_mcp.mcp.gen_client import gen_client

if TYPE_CHECKING:
    from agentis_mcp.core.context import AgentContext

logger = get_logger(__name__)

SEP = "-"


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.
    """

    tool: Tool
    server_name: str
    namespaced_tool_name: str
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}


class ServerAggregator:
    """
    Aggregates multiple MCP servers into a unified interface.
    
    When a client calls a tool, the aggregator routes the call to the appropriate server.
    """

    def __init__(
        self,
        server_names: List[str],
        connection_persistence: bool = False,
        context: Optional["AgentContext"] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the server aggregator.
        
        Args:
            server_names: List of server names to connect to.
            connection_persistence: Whether to maintain persistent connections.
            context: Agent context for server registry and configuration.
            name: Name of this aggregator instance.
        """
        self.server_names = server_names
        self.connection_persistence = connection_persistence
        self.context = context
        self.agent_name = name
        self.initialized = False
        
        # Set up logger with agent name in namespace if available
        self.logger = get_logger(
            f"{__name__}.{name}" if name else __name__
        )
        
        # Tool mapping
        self._namespaced_tool_map: Dict[str, NamespacedTool] = {}
        self._server_to_tool_map: Dict[str, List[NamespacedTool]] = {}
        self._tool_map_lock = Lock()
    
    async def __aenter__(self):
        if self.initialized:
            return self
        
        # Initialize connection manager for persistent connections
        if self.connection_persistence and self.context:
            await self.context.server_registry.connection_manager.__aenter__()
        
        # Load server tools
        await self.load_servers()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """
        Close all persistent connections.
        """
        if self.connection_persistence and self.context:
            try:
                self.logger.info("Shutting down all persistent connections...")
                await self.context.server_registry.connection_manager.disconnect_all()
                self.initialized = False
            finally:
                await self.context.server_registry.connection_manager.__aexit__(None, None, None)
    
    async def load_servers(self):
        """
        Discover tools from each server and build tool mappings.
        """
        if self.initialized:
            self.logger.debug("Server aggregator already initialized.")
            return
        
        if not self.context:
            raise ValueError("Context is required for server aggregator initialization.")
        
        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()
        
        # Create persistent connections if needed
        if self.connection_persistence:
            for server_name in self.server_names:
                self.logger.info(
                    f"Creating persistent connection to server: {server_name}",
                    data={
                        "progress_action": "Starting",
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )
                await self.context.server_registry.connection_manager.get_server(
                    server_name, client_session_factory=AgentisMCPClientSession
                )
        
        async def fetch_tools(client: ClientSession):
            try:
                result: ListToolsResult = await client.list_tools()
                return result.tools or []
            except Exception as e:
                self.logger.error(f"Error loading tools from server", data=e)
                return []
        
        async def load_server_tools(server_name: str):
            tools: List[Tool] = []
            if self.connection_persistence:
                server_connection = (
                    await self.context.server_registry.connection_manager.get_server(
                        server_name, client_session_factory=AgentisMCPClientSession
                    )
                )
                tools = await fetch_tools(server_connection.session)
            else:
                async with gen_client(
                    server_name, 
                    server_registry=self.context.server_registry,
                    client_session_factory=AgentisMCPClientSession
                ) as client:
                    tools = await fetch_tools(client)
            
            return server_name, tools
        
        # Gather tools from all servers concurrently
        results = await gather(
            *(load_server_tools(server_name) for server_name in self.server_names),
            return_exceptions=True,
        )
        
        for result in results:
            if isinstance(result, BaseException):
                continue
                
            server_name, tools = result
            
            self._server_to_tool_map[server_name] = []
            for tool in tools:
                namespaced_tool_name = f"{server_name}{SEP}{tool.name}"
                namespaced_tool = NamespacedTool(
                    tool=tool,
                    server_name=server_name,
                    namespaced_tool_name=namespaced_tool_name,
                )
                
                self._namespaced_tool_map[namespaced_tool_name] = namespaced_tool
                self._server_to_tool_map[server_name].append(namespaced_tool)
            
            self.logger.debug(
                "Server tools loaded",
                data={
                    "progress_action": "Running",
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                    "tools_count": len(tools),
                },
            )
        
        self.initialized = True
    
    async def list_servers(self) -> List[str]:
        """
        Return the list of connected server names.
        """
        if not self.initialized:
            await self.load_servers()
        
        return self.server_names
    
    async def list_tools(self) -> ListToolsResult:
        """
        Return all tools from all servers, with namespaced names.
        """
        if not self.initialized:
            await self.load_servers()
        
        return ListToolsResult(
            tools=[
                namespaced_tool.tool.model_copy(update={"name": namespaced_tool_name})
                for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items()
            ]
        )
    
    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        """
        Call a tool by its namespaced name.
        
        Args:
            name: The tool name, optionally namespaced with server name.
            arguments: Arguments to pass to the tool.
        
        Returns:
            The result of the tool call.
        """
        if not self.initialized:
            await self.load_servers()
        
        server_name: str = None
        local_tool_name: str = None
        
        if SEP in name:  # Namespaced tool name
            parts = name.split(SEP)
            
            for i in range(len(parts) - 1, 0, -1):
                potential_server_name = SEP.join(parts[:i])
                if potential_server_name in self.server_names:
                    server_name = potential_server_name
                    local_tool_name = SEP.join(parts[i:])
                    break
            
            if server_name is None:
                server_name, local_tool_name = name.split(SEP, 1)
        else:
            # Assume un-namespaced, search for matching tool
            for _, tools in self._server_to_tool_map.items():
                for namespaced_tool in tools:
                    if namespaced_tool.tool.name == name:
                        server_name = namespaced_tool.server_name
                        local_tool_name = name
                        break
                if server_name is not None:
                    break
            
            if server_name is None or local_tool_name is None:
                self.logger.error(f"Error: Tool '{name}' not found")
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Tool '{name}' not found")],
                )
        
        self.logger.info(
            "Requesting tool call",
            data={
                "progress_action": "Calling Tool",
                "tool_name": local_tool_name,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )
        
        async def try_call_tool(client: ClientSession):
            try:
                return await client.call_tool(name=local_tool_name, arguments=arguments)
            except Exception as e:
                return CallToolResult(
                    isError=True,
                    content=[
                        TextContent(
                            type="text",
                            text=f"Failed to call tool '{local_tool_name}' on server '{server_name}': {str(e)}",
                        )
                    ],
                )
        
        if self.connection_persistence:
            server_connection = await self.context.server_registry.connection_manager.get_server(
                server_name, client_session_factory=AgentisMCPClientSession
            )
            return await try_call_tool(server_connection.session)
        else:
            self.logger.debug(
                f"Creating temporary connection to server: {server_name}",
                data={
                    "progress_action": "Starting",
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                },
            )
            async with gen_client(
                server_name, 
                server_registry=self.context.server_registry,
                client_session_factory=AgentisMCPClientSession
            ) as client:
                result = await try_call_tool(client)
                self.logger.debug(
                    f"Closing temporary connection to server: {server_name}",
                    data={
                        "progress_action": "Closing",
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )
                return result


class CompoundServer(Server):
    """
    A server that aggregates multiple MCP servers and is itself an MCP server.
    """
    
    def __init__(
        self, 
        server_names: List[str], 
        context: "AgentContext",
        name: str = "CompoundServer"
    ):
        """
        Initialize a compound server.
        
        Args:
            server_names: List of servers to aggregate.
            context: Agent context for configuration and server registry.
            name: Name of this server instance.
        """
        super().__init__(name)
        self.aggregator = ServerAggregator(
            server_names=server_names,
            connection_persistence=True,
            context=context,
            name=name
        )
        
        # Register handlers
        self.list_tools()(self._list_tools)
        self.call_tool()(self._call_tool)
    
    async def _list_tools(self) -> List[Tool]:
        """List all tools from aggregated servers."""
        tools_result = await self.aggregator.list_tools()
        return tools_result.tools
    
    async def _call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        """Call a tool from the aggregated servers."""
        try:
            result = await self.aggregator.call_tool(name=name, arguments=arguments)
            return result.content
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(type="text", text=f"Error calling tool: {str(e)}")
                ],
            )
    
    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport."""
        async with self.aggregator:
            async with stdio_server() as (read_stream, write_stream):
                await self.run(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    initialization_options=self.create_initialization_options(),
                )