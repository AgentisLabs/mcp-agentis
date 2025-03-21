"""
Parallel LLM workflow for the Agentis MCP framework.

This module provides the ParallelLLM class that orchestrates parallel 
processing of requests across multiple agents or functions, and aggregates
the results.
"""

from typing import Any, Callable, List, Optional

from agentis_mcp.agents import Agent
from agentis_mcp.utils.logging import get_logger
from agentis_mcp.workflows.parallel.fan_in import FanIn, FanInInput
from agentis_mcp.workflows.parallel.fan_out import FanOut, FanOutCallable

logger = get_logger(__name__)


class ParallelLLM:
    """
    Parallel LLM workflow for distributing work across multiple agents or functions.
    
    This workflow is useful when:
    1. A task can be divided into subtasks that can be processed in parallel (sectioning)
    2. Multiple perspectives or approaches are needed for higher confidence (voting)
    3. Different components need to work on different aspects of the same data
    
    Examples:
    - Sectioning: Breaking down a long document analysis into sections processed by different agents
    - Voting: Having multiple agents evaluate code for security vulnerabilities
    - Multiple reviews: Having different agents focus on different aspects (grammar, content, style)
    """
    
    def __init__(
        self,
        fan_in_agent: Agent,
        fan_out_agents: Optional[List[Agent]] = None,
        fan_out_functions: Optional[List[FanOutCallable]] = None,
        aggregation_prompt: Optional[str] = None,
    ):
        """
        Initialize the ParallelLLM workflow.
        
        Args:
            fan_in_agent: Agent responsible for aggregating results.
            fan_out_agents: List of agents to use for parallel processing.
            fan_out_functions: List of functions to use for parallel processing.
            aggregation_prompt: Custom prompt for the aggregation phase.
        """
        self.fan_in = FanIn(aggregator_agent=fan_in_agent)
        self.fan_out = FanOut(agents=fan_out_agents, functions=fan_out_functions)
        self.aggregation_prompt = aggregation_prompt
    
    async def run(self, query: str) -> str:
        """
        Run the parallel LLM workflow.
        
        Args:
            query: The query or request to process.
            
        Returns:
            Aggregated result as a string.
        """
        # Fan out the query to multiple agents/functions
        fan_out_results = await self.fan_out.execute(query)
        
        # Fan in the results to a single aggregated result
        aggregated_result = await self.fan_in.aggregate(
            fan_out_results, 
            prompt=self.aggregation_prompt
        )
        
        return aggregated_result