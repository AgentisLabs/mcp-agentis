"""
Fan-out component for parallel processing in the Agentis MCP framework.

This module provides the FanOut class that distributes work to multiple 
parallel tasks or agents.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from agentis_mcp.agents import Agent
from agentis_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Type alias for any callable
FanOutCallable = Callable[[str], Any]


class FanOut:
    """
    Distribute work to multiple parallel tasks or agents.
    
    The FanOut component takes a task and distributes it to multiple agents
    or functions in parallel, then collects the results.
    """
    
    def __init__(
        self,
        agents: Optional[List[Agent]] = None,
        functions: Optional[List[FanOutCallable]] = None,
    ):
        """
        Initialize the FanOut component.
        
        Args:
            agents: List of agents to use for parallel processing.
            functions: List of functions to use for parallel processing.
        """
        self.agents = agents or []
        self.functions = functions or []
        
        if not self.agents and not self.functions:
            raise ValueError("At least one agent or function must be provided")
        
        # Get function metadata for logging
        self.function_names = {}
        for func in self.functions:
            self.function_names[func] = getattr(func, "__name__", str(id(func)))
    
    async def execute(self, input_data: str) -> Dict[str, Any]:
        """
        Execute the input data on all agents and functions in parallel.
        
        Args:
            input_data: The input data to process.
            
        Returns:
            A dictionary mapping agent names or function names to their results.
        """
        tasks = []
        task_ids = []
        
        # Create tasks for all agents
        for agent in self.agents:
            tasks.append(self._execute_agent(agent, input_data))
            task_ids.append(agent.agent_name or str(id(agent)))
        
        # Create tasks for all functions
        for func in self.functions:
            tasks.append(self._execute_function(func, input_data))
            task_ids.append(self.function_names[func])
        
        # Run all tasks in parallel
        logger.info(f"Running {len(tasks)} fan-out tasks")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results to task IDs
        result_dict = {}
        for task_id, result in zip(task_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Error in fan-out task {task_id}: {result}")
                result_dict[task_id] = {"error": str(result)}
            else:
                result_dict[task_id] = result
                
        return result_dict
    
    async def _execute_agent(self, agent: Agent, input_data: str) -> Any:
        """Execute the input data on an agent."""
        try:
            async with agent:
                return await agent.run(input_data)
        except Exception as e:
            logger.error(f"Error executing agent {agent.agent_name}: {e}")
            raise
    
    async def _execute_function(self, func: FanOutCallable, input_data: str) -> Any:
        """Execute the input data on a function."""
        try:
            # Check if the function is async
            if inspect.iscoroutinefunction(func):
                return await func(input_data)
            else:
                # Run synchronous functions in a thread pool
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, func, input_data)
        except Exception as e:
            logger.error(f"Error executing function {self.function_names[func]}: {e}")
            raise