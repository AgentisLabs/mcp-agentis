"""
LLM server for the LLM agent example.

In a real implementation, this would connect to an actual LLM API.
For the example, we'll use a mock implementation.
"""

import asyncio
import os
import sys
from typing import Dict, List, Optional, Any

# Add the package directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp.server import NotificationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server


app = FastMCP("llm-server")


@app.tool
async def generate_completion(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    """
    Generate a completion using an LLM.
    
    Args:
        messages: List of chat messages.
        model: Model to use for generation.
        temperature: Temperature for sampling.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top-p sampling parameter.
        
    Returns:
        Generated completion.
    """
    print(f"Received completion request for model: {model}", file=sys.stderr)
    print(f"Temperature: {temperature}, Top-p: {top_p}", file=sys.stderr)
    print(f"Messages: {len(messages)} message(s)", file=sys.stderr)
    
    # In a real implementation, this would call an actual LLM API
    # For the example, we'll use a mock implementation
    
    # Get the last user message
    user_message = None
    for message in reversed(messages):
        if message.get("role") == "user":
            user_message = message.get("content", "")
            break
    
    if not user_message:
        return {
            "content": "I'm not sure what you're asking about. Could you please clarify?",
            "model": model,
            "finish_reason": "stop"
        }
    
    # Simple pattern-based responses for demonstration
    user_message_lower = user_message.lower()
    
    response = "I'll help you with that!"
    
    # If the user message contains a tool description, simulate LLM recognizing it should use a tool
    if "tool" in user_message_lower and ("search" in user_message_lower or "weather" in user_message_lower):
        response = """I'll help you with that.

I'll use the search tool to find information about the weather:

```json
{"tool": "search_server-search", "parameters": {"query": "weather in San Francisco"}}
```"""
    
    elif "tool" in user_message_lower and "fun fact" in user_message_lower:
        response = """I'll help you with that.

I'll use the fun facts tool to find interesting information:

```json
{"tool": "fun_facts_server-get_fun_fact", "parameters": {"topic": "San Francisco"}}
```"""
    
    elif "tool results" in user_message_lower:
        # If we received tool results, generate a response based on them
        response = """Based on the information I've gathered:

The weather in San Francisco is sunny with a high of 75°F.

Here's a fun fact about San Francisco: The Golden Gate Bridge is actually painted 'International Orange', not gold.

Is there anything else you'd like to know about San Francisco?"""
    
    return {
        "content": response,
        "model": model,
        "finish_reason": "stop"
    }


@app.tool
async def list_available_models() -> List[str]:
    """
    List all available LLM models.
    
    Returns:
        List of available model names.
    """
    return ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]


async def run():
    """Run the LLM server."""
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