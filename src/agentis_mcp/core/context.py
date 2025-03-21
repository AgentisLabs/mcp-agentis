"""
Context management for the Agentis MCP framework.
"""

import asyncio
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from agentis_mcp.config.settings import Settings
from agentis_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Type for human input callback
HumanInputCallback = Callable[[Dict[str, Any]], Any]

# Define type for event handlers
T = TypeVar('T')
EventHandler = Callable[[T], None]


class AgentContext:
    """
    Context object for agent execution.
    
    Provides access to configuration, shared state, and other resources
    needed by agents during execution.
    """

    def __init__(
        self, 
        config: Settings,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_context: Optional["AgentContext"] = None,
        human_input_handler: Optional[HumanInputCallback] = None,
    ):
        """
        Initialize an agent context.
        
        Args:
            config: Configuration settings for the agent.
            agent_name: Name of the agent using this context.
            session_id: Unique identifier for this session.
            parent_context: Parent context for hierarchical context structures.
            human_input_handler: Callback for handling human input requests.
        """
        self.config = config
        self.agent_name = agent_name
        self.session_id = session_id or str(uuid.uuid4())
        self.parent_context = parent_context
        self.state: Dict[str, Any] = {}
        self.event_handlers: Dict[str, List[EventHandler]] = {}
        self.human_input_handler = human_input_handler
        
        # If we have a parent context, inherit its human input handler if we don't have our own
        if parent_context and not human_input_handler:
            self.human_input_handler = parent_context.human_input_handler
        
        # Initialize server registry
        from agentis_mcp.mcp.server_registry import ServerRegistry
        self.server_registry = ServerRegistry(config)
    
    def get_agent_config(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent to get configuration for.
                If None, use the current agent_name.
        
        Returns:
            Agent configuration dictionary.
        """
        name = agent_name or self.agent_name
        if not name:
            return {}
        
        return self.config.agents.get(name, {})
    
    def create_child_context(
        self, 
        agent_name: Optional[str] = None, 
        session_id: Optional[str] = None,
        human_input_handler: Optional[HumanInputCallback] = None,
    ) -> "AgentContext":
        """
        Create a child context with the current context as parent.
        
        Args:
            agent_name: Name of the agent for the child context.
            session_id: Unique identifier for the child session.
            human_input_handler: Custom human input handler for the child context.
        
        Returns:
            New child context.
        """
        return AgentContext(
            config=self.config,
            agent_name=agent_name or self.agent_name,
            session_id=session_id or self.session_id,
            parent_context=self,
            human_input_handler=human_input_handler
        )
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context state.
        
        Args:
            key: State key to retrieve.
            default: Default value if key doesn't exist.
        
        Returns:
            Value from state, or default if not found.
        """
        if key in self.state:
            return self.state[key]
        
        if self.parent_context:
            return self.parent_context.get_state(key, default)
        
        return default
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set a value in the context state.
        
        Args:
            key: State key to set.
            value: Value to store.
        """
        self.state[key] = value
    
    def register_event_handler(self, event_type: str, handler: EventHandler) -> None:
        """
        Register an event handler for a specific event type.
        
        Args:
            event_type: Type of event to handle.
            handler: Function to call when the event occurs.
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    def emit_event(self, event_type: str, event_data: Any) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event_type: Type of event to emit.
            event_data: Data associated with the event.
        """
        # Call local handlers
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
        
        # Propagate to parent context
        if self.parent_context:
            self.parent_context.emit_event(event_type, event_data)
    
    async def request_human_input(
        self, 
        prompt: str, 
        timeout_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Request input from a human user through the registered handler.
        
        Args:
            prompt: The prompt to display to the human.
            timeout_seconds: How long to wait for input before timing out.
            metadata: Additional contextual information.
            
        Returns:
            The human's input response.
            
        Raises:
            RuntimeError: If no human input handler is registered.
            TimeoutError: If the timeout is exceeded.
        """
        if not self.human_input_handler:
            raise RuntimeError("No human input handler registered in context")
        
        request = {
            "request_id": f"human_input_{uuid.uuid4()}",
            "prompt": prompt,
            "timeout_seconds": timeout_seconds,
            "metadata": metadata or {},
            "agent_name": self.agent_name,
            "session_id": self.session_id,
        }
        
        try:
            # Using asyncio.wait_for to handle timeouts
            return await asyncio.wait_for(
                self.human_input_handler(request),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Human input request timed out after {timeout_seconds} seconds")


async def create_root_context(
    config: Settings,
    agent_name: Optional[str] = None,
    session_id: Optional[str] = None,
    human_input_handler: Optional[HumanInputCallback] = None,
) -> AgentContext:
    """
    Create a root agent context with the provided configuration.
    
    Args:
        config: Settings object with configuration.
        agent_name: Optional name for the agent.
        session_id: Optional session identifier.
        human_input_handler: Optional callback for human input.
    
    Returns:
        Initialized AgentContext instance.
    """
    context = AgentContext(
        config=config,
        agent_name=agent_name,
        session_id=session_id,
        human_input_handler=human_input_handler
    )
    
    return context