"""
Base router implementation for directing queries.
"""

import enum
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, TypeVar, Union

from agentis_mcp.agents import Agent
from agentis_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Type for any callable function
CallableFn = Callable[..., Any]

# TypeVar for the router itself
T = TypeVar('T', bound='Router')


class RouteType(str, enum.Enum):
    """Types of routes that can be used by a router."""
    
    AGENT = "agent"
    SERVER = "server"
    FUNCTION = "function"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class RouterResult:
    """Result from a router's routing decision."""
    
    route_type: RouteType
    """Type of the route (agent, server, function)."""
    
    name: str
    """Name of the route."""
    
    score: float
    """Confidence score for this routing decision (0.0 to 1.0)."""
    
    description: str
    """Description of why this route was chosen."""
    
    result: Any
    """The actual route object (agent, server name, function)."""


class Router(Protocol):
    """
    Base protocol for routers that direct queries to appropriate handlers.
    
    A router is responsible for taking a user query and determining which
    agent, server, or function should handle it.
    """
    
    async def route(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request to the appropriate handler.
        
        Args:
            request: The user's query or request.
            top_k: Number of top matches to return.
            
        Returns:
            List of RouterResult objects, sorted by confidence score.
        """
        ...
    
    async def route_to_agent(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request specifically to an agent.
        
        Args:
            request: The user's query or request.
            top_k: Number of top agent matches to return.
            
        Returns:
            List of RouterResult objects for agents, sorted by confidence score.
        """
        ...
    
    async def route_to_server(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request specifically to a server.
        
        Args:
            request: The user's query or request.
            top_k: Number of top server matches to return.
            
        Returns:
            List of RouterResult objects for servers, sorted by confidence score.
        """
        ...
    
    async def route_to_function(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request specifically to a function.
        
        Args:
            request: The user's query or request.
            top_k: Number of top function matches to return.
            
        Returns:
            List of RouterResult objects for functions, sorted by confidence score.
        """
        ...