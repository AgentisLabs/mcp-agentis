"""
Main application class for Agentis MCP framework.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from agentis_mcp.config.settings import Settings, load_config
from agentis_mcp.core.context import AgentContext, create_root_context
from agentis_mcp.utils.logging import get_logger

# Define TypeVar for return type annotation
R = TypeVar('R')


class AgentisApp:
    """
    Main application class that manages context and agent lifecycle.
    
    This is the primary entry point for applications using the Agentis MCP framework.
    It provides a context manager for managing resource lifecycle and access to the
    application context.
    
    Example usage:
        # Create an app instance
        app = AgentisApp()
        
        # Run the app in an async context
        async with app.run() as running_app:
            # App is initialized here
            agent = Agent(context=running_app.context, ...)
            await agent.run("Query to process")
    """
    
    def __init__(
        self,
        name: str = "agentis_app",
        config_path: Optional[str] = None,
        settings: Optional[Settings] = None,
        human_input_handler: Optional[Callable] = None,
    ):
        """
        Initialize the application with a name and optional settings.
        
        Args:
            name: Name of the application.
            config_path: Path to configuration file (if not provided, looks for agentis_mcp.config.yaml).
            settings: Application configuration object (if provided, takes precedence over config_path).
            human_input_handler: Callback for handling human input requests.
        """
        self.name = name
        self._config_path = config_path
        self._settings = settings
        self._human_input_handler = human_input_handler
        
        self._logger = None
        self._context = None
        self._initialized = False
        self._session_id = None
    
    @property
    def context(self) -> AgentContext:
        """Get the current application context."""
        if self._context is None:
            raise RuntimeError(
                "AgentisApp not initialized. Please call initialize() first, or use async with app.run()."
            )
        return self._context
    
    @property
    def config(self) -> Settings:
        """Get the current application configuration."""
        return self.context.config
    
    @property
    def server_registry(self):
        """Get the server registry from the context."""
        return self.context.server_registry
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id
    
    @property
    def logger(self):
        """Get the application logger."""
        if self._logger is None:
            self._logger = get_logger(f"agentis.{self.name}")
        return self._logger
    
    async def initialize(self):
        """Initialize the application and context."""
        if self._initialized:
            return
        
        # Generate a session ID if we don't have one
        if not self._session_id:
            self._session_id = str(uuid.uuid4())
        
        # Load configuration if needed
        if self._settings is None:
            config = load_config(self._config_path)
        else:
            config = self._settings
        
        # Create the root context
        self._context = await create_root_context(
            config=config,
            agent_name=self.name,
            session_id=self._session_id,
            human_input_handler=self._human_input_handler
        )
        
        self._initialized = True
        self.logger.info(f"AgentisApp initialized - app_name: {self.name}, session_id: {self._session_id}")
    
    async def cleanup(self):
        """Clean up application resources."""
        if not self._initialized:
            return
        
        self.logger.info(f"AgentisApp cleaning up - app_name: {self.name}, session_id: {self._session_id}")
        
        # Clean up any resources
        # In the future, we might need to handle connection cleanup here
        
        self._context = None
        self._initialized = False
    
    @asynccontextmanager
    async def run(self):
        """
        Run the application as an async context manager.
        
        Example:
            async with app.run() as running_app:
                # App is initialized here
                pass
        
        Yields:
            The initialized application instance.
        """
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
    
    def register_human_input_handler(self, handler: Callable):
        """
        Register a callback function to handle human input requests.
        
        Args:
            handler: Callback function that handles human input requests.
                    Should accept a request dictionary and return a response.
        """
        if self._initialized and self._context:
            self._context.human_input_handler = handler
        else:
            self._human_input_handler = handler
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """
        Register an event handler for a specific event type.
        
        Args:
            event_type: Type of event to handle.
            handler: Function to call when the event occurs.
        """
        if self._initialized and self._context:
            self._context.register_event_handler(event_type, handler)
        else:
            # Store for later registration once context is initialized
            async def register_later():
                await self.initialize()
                self._context.register_event_handler(event_type, handler)
            
            asyncio.create_task(register_later())