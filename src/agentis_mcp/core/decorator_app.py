"""
Decorator-based interface for Agentis MCP applications.

This module provides a simplified way to create and use agents using decorators.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

import yaml

from agentis_mcp.agents import Agent, LLMConfig
from agentis_mcp.app import AgentisApp
from agentis_mcp.config.settings import Settings, load_config


class AgentisDecorator:
    """
    A decorator-based interface for Agentis MCP applications.
    
    This class provides a simplified way to create and manage agents using
    Python decorators, hiding much of the complexity of agent initialization.
    
    Example:
        ```python
        # Create the application
        app = AgentisDecorator("my-app")
        
        # Define an agent using decorators
        @app.agent(
            name="my_agent",
            instruction="You are a helpful assistant.",
            servers=["fetch", "filesystem"]
        )
        async def my_function():
            # Use the app's context manager
            async with app.run() as agents:
                # Send a message to the agent
                result = await agents.send("my_agent", "What files are in this directory?")
                print(result)
        ```
    """
    
    def __init__(
        self, 
        name: str, 
        config_path: Optional[str] = None,
        llm_provider: str = "mock",
    ):
        """
        Initialize the decorator interface.
        
        Args:
            name: Name of the application.
            config_path: Optional path to configuration file.
            llm_provider: Default LLM provider for agents (mock, openai, anthropic).
        """
        self.name = name
        self.config_path = config_path
        self.llm_provider = llm_provider
        
        # Load configuration
        self.config = load_config(config_path) if config_path else None
        
        # Create app
        self.app = AgentisApp(
            name=name,
            config_path=config_path,
            settings=self.config,
        )
        
        # Store agent configurations for later instantiation
        self.agents: Dict[str, Dict[str, Any]] = {}
    
    def agent(
        self, 
        name: str, 
        instruction: str, 
        servers: List[str],
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator to create and register an agent.
        
        Args:
            name: Name of the agent.
            instruction: Base instruction for the agent.
            servers: List of server names the agent should connect to.
            llm_config: Optional LLM configuration for the agent.
            
        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            # Store the agent configuration for later instantiation
            self.agents[name] = {
                "instruction": instruction, 
                "servers": servers,
                "llm_config": llm_config or {"provider": self.llm_provider},
            }
            
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
                
            return wrapper
            
        return decorator
    
    @asynccontextmanager
    async def run(self):
        """
        Context manager for running the application.
        
        Handles setup and teardown of the app and agents.
        
        Yields:
            AgentWrapper: A wrapper object providing a simplified interface to agents.
        """
        async with self.app.run() as app_instance:
            active_agents = {}
            agent_contexts = []
            
            # Create all registered agents
            for name, config in self.agents.items():
                # Create agent
                agent = Agent(
                    context=app_instance.context,
                    agent_name=name,
                    instruction=config["instruction"],
                    server_names=config["servers"],
                    llm_config=LLMConfig(**config["llm_config"]),
                )
                active_agents[name] = agent
                
                # Initialize agent
                ctx = await agent.__aenter__()
                agent_contexts.append((agent, ctx))
            
            # Create wrapper with simplified interface
            wrapper = AgentWrapper(app_instance, active_agents)
            
            try:
                yield wrapper
            finally:
                # Clean up agents
                for agent, _ in agent_contexts:
                    await agent.__aexit__(None, None, None)


class AgentWrapper:
    """
    Wrapper class providing a simplified interface to agents.
    """
    
    def __init__(self, app: AgentisApp, agents: Dict[str, Agent]):
        """
        Initialize the agent wrapper.
        
        Args:
            app: The AgentisApp instance.
            agents: Dictionary of agent instances.
        """
        self.app = app
        self.agents = agents
    
    async def send(self, agent_name: str, message: str) -> str:
        """
        Send a message to a specific agent and get the response.
        
        Args:
            agent_name: Name of the agent to send message to.
            message: Message to send.
            
        Returns:
            Agent's response.
            
        Raises:
            ValueError: If the agent is not found.
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        result = await agent.run(message)
        return result
    
    def get_agent(self, agent_name: str) -> Agent:
        """
        Get an agent by name.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            Agent instance.
            
        Raises:
            ValueError: If the agent is not found.
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        return self.agents[agent_name]