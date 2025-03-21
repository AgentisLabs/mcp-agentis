"""
Simple search server for demo purposes.
"""

import asyncio
import os
import sys

# Add the package directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp.server import NotificationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server


app = FastMCP("search-server")


@app.tool
async def search(query: str) -> str:
    """
    Search for information.
    
    Args:
        query: The search query.
        
    Returns:
        Results of the search.
    """
    # This is a mock implementation
    print(f"Received search query: {query}", file=sys.stderr)
    
    if "weather" in query.lower():
        return "The weather is sunny with a high of 75°F."
    elif "news" in query.lower():
        return "Latest news: Agentis MCP framework released!"
    else:
        return f"Search results for: {query}\n- Result 1\n- Result 2\n- Result 3"


@app.tool
async def quick_facts(topic: str) -> str:
    """
    Get quick facts about a topic.
    
    Args:
        topic: The topic to get facts about.
        
    Returns:
        Quick facts about the topic.
    """
    # This is a mock implementation
    print(f"Received quick facts request for: {topic}", file=sys.stderr)
    
    facts = {
        "python": "Python is a programming language created by Guido van Rossum in 1991.",
        "mcp": "MCP (Machine Conversation Protocol) is a protocol for communication between AI agents and tools.",
        "agentis": "Agentis MCP is a framework for building AI agents with MCP server connectivity."
    }
    
    return facts.get(topic.lower(), f"No facts available for {topic}")


async def run():
    """Run the search server."""
    async with stdio_server() as (read_stream, write_stream):
        await app._mcp_server.run(
            read_stream,
            write_stream,
            app._mcp_server.create_initialization_options(
                notification_options=NotificationOptions(
                    tools_changed=True, resources_changed=True
                )
            ),
        )


if __name__ == "__main__":
    asyncio.run(run())