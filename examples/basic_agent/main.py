"""
Basic agent example for Agentis MCP framework.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the package directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentis_mcp import AgentContext, Agent, load_config


class BasicAgent(Agent):
    """Simple agent implementation."""
    
    async def run(self, query: str, **kwargs) -> Any:
        """
        Process a user query.
        
        Args:
            query: The user's query.
            
        Returns:
            Response to the query.
        """
        self.logger.info(f"Processing query: {query}")
        
        # List available tools
        tools = await self.list_tools()
        self.logger.info(f"Available tools: {len(tools)}")
        
        # Basic processing: just pass the query to the search tool if available
        for tool in tools:
            if "search" in tool["name"].lower():
                try:
                    result = await self.call_tool(tool["name"], {"query": query})
                    return {
                        "result": result,
                        "tool_used": tool["name"]
                    }
                except Exception as e:
                    self.logger.error(f"Error calling search tool: {e}")
        
        # If no search tool, return a simple response
        return {
            "result": f"I understood your query: '{query}', but I don't have a search tool available.",
            "tool_used": None
        }


async def main():
    """Run the basic agent example."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "agentis_mcp.config.yaml")
    config = load_config(config_path)
    
    # Create context
    context = AgentContext(config, agent_name="basic_agent")
    
    # Create and run agent
    async with BasicAgent(context) as agent:
        # Process a query
        result = await agent.run("What is the weather in San Francisco?")
        print("\nResult:")
        print(f"  {result['result']}")
        if result['tool_used']:
            print(f"  Tool used: {result['tool_used']}")
        
        # Call a specific tool if available
        tools = await agent.list_tools()
        if tools:
            print("\nAvailable tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    asyncio.run(main())