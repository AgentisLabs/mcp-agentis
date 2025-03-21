"""
Fan-in component for parallel processing in the Agentis MCP framework.

This module provides the FanIn class that aggregates results from multiple 
parallel tasks or agents.
"""

from typing import Any, Dict, List, Optional, Union

from agentis_mcp.agents import Agent
from agentis_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Type alias for results from fan-out operations
FanInInput = Dict[str, Any]


class FanIn:
    """
    Aggregate results from multiple parallel tasks or agents.
    
    The FanIn component takes results from multiple tasks or agents
    and aggregates them into a single result, typically using an LLM
    to perform the aggregation.
    """
    
    def __init__(
        self,
        aggregator_agent: Agent,
    ):
        """
        Initialize the FanIn component.
        
        Args:
            aggregator_agent: Agent responsible for aggregating results.
        """
        self.aggregator_agent = aggregator_agent
    
    async def aggregate(self, results: FanInInput, prompt: Optional[str] = None) -> str:
        """
        Aggregate results from multiple tasks or agents.
        
        Args:
            results: Dictionary of results from fan-out operations.
            prompt: Optional custom prompt to guide aggregation.
            
        Returns:
            Aggregated result as a string.
        """
        # Format the results for the aggregator agent
        formatted_results = self._format_results(results)
        
        # Create a prompt for the aggregator agent
        if prompt:
            aggregation_prompt = f"{prompt}\n\n{formatted_results}"
        else:
            aggregation_prompt = (
                f"You are tasked with aggregating the following results from multiple sources. "
                f"Analyze the results, identify common themes, and provide a comprehensive summary "
                f"that combines the insights from all sources.\n\n{formatted_results}"
            )
        
        # Run the aggregator agent
        async with self.aggregator_agent:
            aggregated_result = await self.aggregator_agent.run(aggregation_prompt)
        
        return aggregated_result
    
    def _format_results(self, results: FanInInput) -> str:
        """
        Format the results from multiple sources for the aggregator agent.
        
        Args:
            results: Dictionary of results from fan-out operations.
            
        Returns:
            Formatted results as a string.
        """
        formatted_sections = []
        
        for source_name, result in results.items():
            # Handle error cases
            if isinstance(result, dict) and 'error' in result:
                formatted_sections.append(
                    f"Source: {source_name}\nError: {result['error']}\n"
                )
                continue
                
            # Format the result based on its type
            if isinstance(result, str):
                formatted_sections.append(f"Source: {source_name}\n{result}\n")
            elif isinstance(result, dict):
                # Try to prettify the dictionary
                formatted_result = "\n".join(f"  {k}: {v}" for k, v in result.items())
                formatted_sections.append(f"Source: {source_name}\n{formatted_result}\n")
            elif isinstance(result, list):
                # Try to prettify the list
                formatted_result = "\n".join(f"  - {item}" for item in result)
                formatted_sections.append(f"Source: {source_name}\n{formatted_result}\n")
            else:
                # Fallback for any other type
                formatted_sections.append(f"Source: {source_name}\n{result}\n")
        
        return "\n".join(formatted_sections)